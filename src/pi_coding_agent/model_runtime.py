"""中央模型/认证运行时（对齐 TS core/model-runtime.ts）。

ModelRuntime 是 coding-agent 的模型基础设施：

- 组合 provider：内置 provider + models.json 覆盖 + 扩展覆盖；
- 认证解析：存储凭证 > models.json 配置 key > 环境变量；
- 可用性快照：available / configuredProviders / auth checks；
- 动态刷新：Models.refresh + 组合 provider 重建；
- 实现 pi_ai Models 的流式接口（stream/complete）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from pi_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    Context,
    DeferredHandle,
    Model,
    ModelCompat,
    Models,
    Provider,
    ModelCost,
    ModelInput,
    StreamOptions,
)
from pi_ai.types.common import ModelThinkingLevel
from pi_ai.api.api_provider_registry import get_api_provider, invoke_api_stream
from pi_ai.auth import ApiKeyCredential, EnvApiKeyAuth, ResolvedAuth
from pi_ai.auth.resolve import ModelsError
from pi_ai.auth.types import AuthResult, ModelAuth, credential_type
from pi_ai.models.models_store import InMemoryModelsStore, ModelsStore
from pi_ai.models import ModelsRefreshOptions, ModelsRefreshResult
from pi_ai.provider import _API_KIND_IDS

from .auth_storage import AuthStorage
from .model_config import (
    CostConfig,
    ModelConfig,
    ModelDefinition,
    ModelOverride,
    ProviderOverride,
)
from .resolve_config_value import (
    get_config_value_env_var_names,
    is_command_config_value,
    resolve_config_value_or_throw,
    resolve_headers_or_throw,
)
from .runtime_credentials import RuntimeCredentials


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------


class AuthCheck(TypedDict):
    type: str
    source: str


class ModelRuntimeAuthOverrides(TypedDict, total=False):
    api_key: str
    env: dict[str, str]
    min_oauth_validity_ms: int


class AuthStatus(TypedDict):
    configured: bool
    source: str | None


class CompatibilityRequestConfig(TypedDict, total=False):
    headers: dict[str, str]
    auth_header: bool


class ProviderConfigInput(dict):
    """扩展 provider 配置（keys: name/base_url/api_key/api/headers/
    auth_header/models/oauth/stream_simple/refresh_models）。"""


# ---------------------------------------------------------------------------
# 组合认证（base + models.json + 扩展）
# ---------------------------------------------------------------------------


def _credential_key(credential: Any) -> str | None:
    if credential is None:
        return None
    if isinstance(credential, dict):
        return credential.get("key")
    return getattr(credential, "key", None)


@dataclass(slots=True)
class ComposedApiKeyAuth:
    """合并内置 provider + models.json + 扩展的 API Key 认证。

    解析优先级（与 TS composeApiKeyAuth 一致）：
    存储凭证 > models.json/扩展配置 key > 内置环境变量。
    """

    display_name: str
    base: EnvApiKeyAuth | None
    configured_key: str | None
    env_vars: list[str]

    def resolve(
        self,
        credential: ApiKeyCredential | dict | None = None,
    ) -> ResolvedAuth | None:
        key = _credential_key(credential)
        if key:
            return ResolvedAuth(api_key=key, source="stored credential")
        if self.configured_key is not None:
            try:
                resolved = resolve_config_value_or_throw(
                    self.configured_key,
                    f'API key for provider "{self.display_name}"',
                )
            except ValueError:
                return None
            return ResolvedAuth(api_key=resolved, source="configured API key")
        if self.base is not None:
            return self.base.resolve(credential)
        return None


# ---------------------------------------------------------------------------
# 组合辅助
# ---------------------------------------------------------------------------


def _merge_compat(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not override:
        return base
    merged = dict(base or {})
    merged.update(override)
    for key in ("openRouterRouting", "vercelGatewayRouting", "chatTemplateKwargs"):
        base_value = (base or {}).get(key)
        override_value = override.get(key)
        if isinstance(base_value, dict) or isinstance(override_value, dict):
            merged[key] = {**(base_value or {}), **(override_value or {})}
    return merged


def _merge_cost(model: Model, cost: CostConfig | None) -> Any:
    if cost is None:
        return model.cost
    from pi_ai import ModelCost, ModelCostTier

    tiers = [
        ModelCostTier(
            input=tier.input,
            output=tier.output,
            cache_read=tier.cache_read,
            cache_write=tier.cache_write,
            input_tokens_above=tier.input_tokens_above,
        )
        for tier in cost.tiers
    ]
    return ModelCost(
        input=cost.input,
        output=cost.output,
        cache_read=cost.cache_read,
        cache_write=cost.cache_write,
        tiers=tiers,
    )


def _apply_model_override(model: Model, override: ModelOverride) -> Model:
    return Model(
        id=model.id,
        provider=model.provider,
        api=model.api,
        name=override.name if override.name is not None else model.name,
        aliases=(list(override.aliases) if override.aliases is not None else list(model.aliases)),
        input=cast(list[ModelInput], list(override.input))
        if override.input is not None
        else model.input,
        output=list(model.output),
        cost=_merge_cost(model, override.cost),
        max_tokens=override.max_tokens if override.max_tokens is not None else model.max_tokens,
        base_url=model.base_url,
        context_window=(
            override.context_window if override.context_window is not None else model.context_window
        ),
        headers=override.headers if override.headers is not None else model.headers,
        compat=cast(
            ModelCompat,
            _merge_compat(
                cast(dict[str, Any] | None, model.compat),
                override.compat,
            ),
        ),
        thinking_level_map=(
            {
                **(model.thinking_level_map or {}),
                **cast(dict[ModelThinkingLevel, str | None], override.thinking_level_map or {}),
            }
            if override.thinking_level_map is not None
            else model.thinking_level_map
        ),
        reasoning=override.reasoning if override.reasoning is not None else model.reasoning,
    )


def _model_from_json(
    provider_id: str,
    definition: ModelDefinition,
    provider_config: ProviderOverride | None,
    defaults: Model | None,
) -> Model:
    api = (
        definition.api
        or (provider_config.api if provider_config else None)
        or (defaults.api if defaults else None)
    )
    if not api:
        raise ValueError(
            f'Provider {provider_id}, model {definition.id}: no "api" specified. '
            "Set at provider or model level."
        )
    base_url = (
        definition.base_url
        or (provider_config.base_url if provider_config else None)
        or (defaults.base_url if defaults else None)
    )
    if not base_url:
        raise ValueError(
            f'Provider {provider_id}: "baseUrl" is required when defining custom models.'
        )
    cost = definition.cost
    return Model(
        id=definition.id,
        provider=provider_id,
        api=api,
        name=definition.name or definition.id,
        aliases=list(definition.aliases or []),
        input=cast(list[ModelInput], list(definition.input))
        if definition.input is not None
        else ["text"],
        output=["text"],
        cost=cast(
            ModelCost,
            _cost_config_to_model_cost(cost)
            if cost is not None
            else (defaults.cost if defaults else None),
        ),
        max_tokens=definition.max_tokens if definition.max_tokens is not None else 16384,
        base_url=base_url,
        context_window=definition.context_window
        if definition.context_window is not None
        else 128000,
        headers=definition.headers,
        compat=cast(
            ModelCompat,
            _merge_compat(
                cast(dict[str, Any] | None, provider_config.compat if provider_config else None),
                definition.compat,
            ),
        ),
        thinking_level_map=cast(
            dict[ModelThinkingLevel, str | None] | None,
            definition.thinking_level_map,
        ),
        reasoning=bool(definition.reasoning),
    )


def _cost_config_to_model_cost(cost: CostConfig):
    from pi_ai import ModelCost, ModelCostTier

    return ModelCost(
        input=cost.input,
        output=cost.output,
        cache_read=cost.cache_read,
        cache_write=cost.cache_write,
        tiers=[
            ModelCostTier(
                input=tier.input,
                output=tier.output,
                cache_read=tier.cache_read,
                cache_write=tier.cache_write,
                input_tokens_above=tier.input_tokens_above,
            )
            for tier in cost.tiers
        ],
    )


def _apply_models_json(
    provider_id: str,
    base_models: list[Model],
    config: ProviderOverride | None,
) -> list[Model]:
    if config is None:
        return list(base_models)
    if config.oauth and not config.base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required when "oauth" is set.')
    has_overrides = bool(config.model_overrides)
    if (
        not config.models
        and not config.base_url
        and not config.headers
        and not config.compat
        and not has_overrides
        and not config.api_key
        and not config.oauth
        and config.auth_header is None
    ):
        raise ValueError(
            f'Provider {provider_id}: must specify "baseUrl", "headers", "compat", '
            '"modelOverrides", or "models".'
        )

    models: list[Model] = [
        Model(
            id=model.id,
            provider=model.provider,
            api=model.api,
            name=model.name,
            aliases=list(model.aliases),
            input=list(model.input),
            output=list(model.output),
            cost=model.cost,
            max_tokens=model.max_tokens,
            base_url=config.base_url or model.base_url,
            context_window=model.context_window,
            headers=model.headers,
            compat=cast(
                ModelCompat,
                _merge_compat(
                    cast(dict[str, Any] | None, model.compat),
                    config.compat,
                ),
            ),
            thinking_level_map=model.thinking_level_map,
            reasoning=model.reasoning,
        )
        for model in base_models
    ]
    for definition in config.models:
        existing_index = next(
            (index for index, model in enumerate(models) if model.id == definition.id),
            -1,
        )
        defaults = (
            models[existing_index] if existing_index >= 0 else (models[0] if models else None)
        )
        model = _model_from_json(provider_id, definition, config, defaults)
        if existing_index >= 0:
            models[existing_index] = model
        else:
            models.append(model)
    return models


def _apply_extension(
    provider_id: str,
    models: list[Model],
    extension: ProviderConfigInput | None,
) -> list[Model]:
    if not extension:
        return list(models)
    definitions = extension.get("models")
    if not definitions:
        base_url = extension.get("base_url")
        if base_url:
            return [
                Model(
                    id=model.id,
                    provider=model.provider,
                    api=model.api,
                    name=model.name,
                    aliases=list(model.aliases),
                    input=list(model.input),
                    output=list(model.output),
                    cost=model.cost,
                    max_tokens=model.max_tokens,
                    base_url=base_url,
                    context_window=model.context_window,
                    headers=model.headers,
                    compat=model.compat,
                    thinking_level_map=model.thinking_level_map,
                    reasoning=model.reasoning,
                )
                for model in models
            ]
        return list(models)

    result: list[Model] = []
    extension_api = extension.get("api")
    extension_base_url = extension.get("base_url")
    for definition in definitions:
        defaults = next(
            (model for model in models if model.id == definition.get("id")),
            models[0] if models else None,
        )
        api = definition.get("api") or extension_api or (defaults.api if defaults else None)
        if not api:
            raise ValueError(
                f'Provider {provider_id}, model {definition.get("id")}: no "api" specified. '
                "Set at provider or model level."
            )
        base_url = (
            definition.get("base_url")
            or extension_base_url
            or (defaults.base_url if defaults else None)
        )
        if not base_url:
            raise ValueError(
                f'Provider {provider_id}: "baseUrl" is required when defining custom models.'
            )
        result.append(
            Model(
                id=definition.get("id"),
                provider=provider_id,
                api=api,
                name=definition.get("name") or definition.get("id"),
                aliases=list(definition.get("aliases") or []),
                input=list(definition.get("input") or ["text"]),
                output=["text"],
                cost=cast(
                    ModelCost,
                    definition.get("cost")
                    if isinstance(definition.get("cost"), dict)
                    else (defaults.cost if defaults else None),
                ),
                max_tokens=definition.get("max_tokens", 16384),
                base_url=base_url,
                context_window=definition.get("context_window", 128000),
                headers=definition.get("headers"),
                compat=definition.get("compat"),
                thinking_level_map=definition.get("thinking_level_map"),
                reasoning=bool(definition.get("reasoning", False)),
            )
        )
    return result


def _configured_api_key(
    config: ProviderOverride | None,
    extension: ProviderConfigInput | None,
) -> str | None:
    if extension is not None and extension.get("api_key") is not None:
        return extension["api_key"]
    return config.api_key if config is not None else None


def _configured_headers(
    config: ProviderOverride | None,
    extension: ProviderConfigInput | None,
) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if config is not None and config.headers:
        headers.update(config.headers)
    if extension is not None and extension.get("headers"):
        headers.update(extension["headers"])
    return headers or None


def _compose_models(
    provider_id: str,
    base: Provider | None,
    config: ProviderOverride | None,
    extension: ProviderConfigInput | None,
) -> list[Model]:
    models = _apply_models_json(provider_id, list(base.get_models()) if base else [], config)
    models = _apply_extension(provider_id, models, extension)
    overrides = config.model_overrides if config is not None else {}
    return [
        _apply_model_override(model, overrides[model.id]) if model.id in overrides else model
        for model in models
    ]


def _compose_api_key_auth(
    provider_id: str,
    base: Provider | None,
    config: ProviderOverride | None,
    extension: ProviderConfigInput | None,
) -> ComposedApiKeyAuth | None:
    base_auth = base.auth if base is not None else None
    inherited = base_auth if isinstance(base_auth, EnvApiKeyAuth) else None
    raw_key = _configured_api_key(config, extension)
    if inherited is None and raw_key is None:
        # 无认证方法：本地 provider（auth=None）或纯扩展 oauth 交给上层处理。
        return None

    env_vars: list[str] = []
    if inherited is not None:
        env_vars.extend(inherited.env_vars)
    for value in (raw_key,):
        if value is not None:
            for name in get_config_value_env_var_names(value):
                if name not in env_vars:
                    env_vars.append(name)
    return ComposedApiKeyAuth(
        display_name=provider_id,
        base=inherited,
        configured_key=raw_key,
        env_vars=env_vars,
    )


# ---------------------------------------------------------------------------
# ModelRuntime
# ---------------------------------------------------------------------------


class ModelRuntime:
    """中央模型/认证运行时（实现 pi_ai Models 流式接口）。"""

    def __init__(
        self,
        models: Models,
        auth_store: AuthStorage,
        config: ModelConfig | None = None,
        *,
        models_path: str | None = None,
        models_store: ModelsStore | None = None,
    ) -> None:
        self._models = models
        self._auth_store = auth_store
        self._config = config or ModelConfig()
        self._models_path = models_path
        self._models_store = models_store or InMemoryModelsStore()

        # 让 Models 使用运行时凭证层（runtime API key 覆盖 + auth.json）。
        self._credentials = RuntimeCredentials(auth_store)
        self._models._credentials = self._credentials
        for provider in self._models.get_providers():
            provider._credential_store = cast(Any, self._credentials)

        self._builtins: dict[str, Provider] = {
            provider.id: provider for provider in self._models.get_providers()
        }
        self._native_extension_providers: dict[str, Provider] = {}
        self._extension_providers: dict[str, ProviderConfigInput] = {}
        self._composition_errors: dict[str, str] = {}

        self._available: list[Model] = []
        self._configured_providers: set[str] = set()
        self._stored_providers: set[str] = set()
        self._auth_checks: dict[str, AuthCheck | None] = {}
        self._availability_error: str | None = None
        self._availability_refresh: asyncio.Task | None = None

        self.rebuild_providers()

    @property
    def auth_store(self) -> AuthStorage:
        return self._auth_store

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @staticmethod
    async def create(
        *,
        providers: list[Provider] | None = None,
        auth_path: str | None = None,
        auth_store: AuthStorage | None = None,
        models_path: str | None = None,
        models_store: ModelsStore | None = None,
        allow_model_network: bool = False,
        model_refresh_timeout_ms: int = 15000,
    ) -> "ModelRuntime":
        """构建运行时：可选 create-time 网络刷新（默认离线）。"""
        store = auth_store or AuthStorage.create(auth_path)
        models = Models(credentials=store)
        for provider in providers or []:
            models.add_provider(provider)
        config = await ModelConfig.load(models_path)
        runtime = ModelRuntime(
            models,
            store,
            config=config,
            models_path=models_path,
            models_store=models_store,
        )
        # 始终先计算可用性快照（不依赖网络；auth.json/环境变量即时生效）。
        await runtime._run_availability_refresh()
        if allow_model_network:

            async def _refresh_with_timeout() -> None:
                try:
                    await asyncio.wait_for(
                        runtime.refresh(ModelsRefreshOptions(allow_network=True)),
                        timeout=model_refresh_timeout_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    pass

            await _refresh_with_timeout()
        return runtime

    # ------------------------------------------------------------------
    # Provider 管理
    # ------------------------------------------------------------------

    def _provider_ids(self) -> set[str]:
        return (
            set(self._builtins)
            | set(self._native_extension_providers)
            | set(self._config.get_provider_ids())
            | set(self._extension_providers)
        )

    def compose_model_provider(self, provider_id: str) -> Provider:
        """合并基础 provider + models.json 覆盖 + 扩展覆盖。"""
        base = self._native_extension_providers.get(provider_id) or self._builtins.get(provider_id)
        extension = self._extension_providers.get(provider_id)
        config = self._config.get_provider_override(provider_id)
        if base is None and config is None and extension is None:
            raise ValueError(f"Unknown provider: {provider_id}")

        if base is not None and config is None and extension is None:
            # 无覆盖：原样使用内置 provider（保持其 auth/stream 行为）。
            return base

        models = _compose_models(provider_id, base, config, extension)
        api_key_auth = _compose_api_key_auth(provider_id, base, config, extension)
        auth = api_key_auth
        if auth is None and not (base is not None and base.auth is None):
            # 无认证方法：自定义 provider 必须配置 apiKey。
            raise ValueError(f"Provider {provider_id}: no authentication method configured.")

        base_url = (
            extension.get("base_url")
            if extension is not None and extension.get("base_url") is not None
            else (config.base_url if config is not None else None)
        ) or (base.base_url if base is not None else None)
        name = (
            (
                extension.get("name")
                if extension is not None and extension.get("name") is not None
                else (config.name if config is not None else None)
            )
            or (base.name if base is not None else None)
            or provider_id
        )

        raw_headers = _configured_headers(config, extension)
        auth_header = bool(
            extension.get("auth_header")
            if extension is not None and extension.get("auth_header") is not None
            else (config.auth_header if config is not None else None)
        )
        has_model_headers = any(model.headers for model in models)
        needs_wrapper = bool(raw_headers) or auth_header or has_model_headers

        stream_fn = (
            self._build_composed_stream_fn(
                provider_id,
                base,
                extension,
                base_url,
            )
            if needs_wrapper
            else None
        )
        refresh_models = self._build_composed_refresh_models(base, extension)
        return Provider(
            id=provider_id,
            name=name,
            auth=cast(Any, auth),
            models=models,
            _api_kind=base._api_kind if base is not None else "completions",
            base_url=base_url,
            _stream_fn=stream_fn,
            refresh_models=refresh_models,
        )

    def _build_composed_stream_fn(
        self,
        provider_id: str,
        base: Provider | None,
        extension: ProviderConfigInput | None,
        composed_base_url: str | None,
    ):
        """构建注入 headers/authHeader 的流包装。"""

        async def _stream(
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            opts = dict(options or {})
            resolution = await self.get_auth(
                model,
                cast(
                    ModelRuntimeAuthOverrides,
                    {"api_key": opts.get("api_key"), "env": opts.get("env")},
                ),
            )
            if resolution is None:
                raise ModelsError("auth", f"Provider is not configured: {model.provider}")
            auth = resolution.auth
            if opts.get("api_key") is None and auth.get("api_key"):
                opts["api_key"] = auth["api_key"]
            headers = {
                **(auth.get("headers") or {}),
                **(cast(dict[str, str | None], opts.get("headers") or {})),
            }
            if headers:
                opts["headers"] = headers
            opts["base_url"] = opts.get("base_url") or composed_base_url or ""

            if extension is not None and extension.get("stream_simple") is not None:
                return await extension["stream_simple"](model, context, opts)
            if base is not None and base._stream_fn is not None:
                return await base._stream_fn(model, context, cast(StreamOptions, opts))
            api_id = model.api or (
                _API_KIND_IDS.get(base._api_kind, base._api_kind)
                if base is not None
                else "openai-completions"
            )
            entry = get_api_provider(api_id)
            if entry is None:
                raise ValueError(
                    f"No API provider registered for api: {api_id} (provider api kind: {api_id})"
                )
            return await invoke_api_stream(entry.stream, model, context, cast(StreamOptions, opts))

        return _stream

    def _build_composed_refresh_models(self, base: Provider | None, extension):
        """组合 refreshModels：先跑 base，再跑扩展（若有）。"""
        base_refresh = base.refresh_models if base is not None else None
        extension_refresh = extension.get("refresh_models") if extension else None
        if base_refresh is None and extension_refresh is None:
            return None

        async def _refresh(context) -> None:
            if base_refresh is not None:
                await base_refresh(context)
            if extension_refresh is not None:
                await extension_refresh(context)

        return _refresh

    def register_native_provider(self, provider: Provider) -> None:
        """注册原生扩展 provider（覆盖 models.json / 内置同 id 的 provider）。"""
        if not provider.id.strip():
            raise ValueError("Provider id must not be empty.")
        self._extension_providers.pop(provider.id, None)
        self._native_extension_providers[provider.id] = provider
        self._recompose_provider(provider.id)
        self._update_model_snapshot()
        self._schedule_refresh()

    def register_provider(self, provider_id: str, config: ProviderConfigInput) -> None:
        """注册扩展 provider 配置（重复注册时保留未定义字段）。"""
        base = self._native_extension_providers.get(provider_id) or self._builtins.get(provider_id)
        models_config = self._config.get_provider_override(provider_id)
        _validate_extension_provider(provider_id, base, models_config, config)
        self._native_extension_providers.pop(provider_id, None)
        previous = self._extension_providers.get(provider_id)
        effective = dict(previous or {})
        for key, value in config.items():
            if value is not None:
                effective[key] = value
        self._extension_providers[provider_id] = cast(ProviderConfigInput, effective)
        self._recompose_provider(provider_id)
        self._update_model_snapshot()
        # 已配置时先给一个临时可用性条目，异步 refresh 会修正。
        check = self._provisional_auth_check(
            provider_id, cast(ProviderConfigInput | None, effective)
        )
        if check is not None:
            self._auth_checks[provider_id] = check
            self._configured_providers.add(provider_id)
            self._available = [
                model
                for model in self._models.get_models()
                if model.provider in self._configured_providers
            ]
        self._schedule_refresh()

    def unregister_provider(self, provider_id: str) -> None:
        self._extension_providers.pop(provider_id, None)
        self._native_extension_providers.pop(provider_id, None)
        self._recompose_provider(provider_id)
        self._update_model_snapshot()
        self._schedule_refresh()

    def _provisional_auth_check(
        self, provider_id: str, extension: ProviderConfigInput | None
    ) -> AuthCheck | None:
        if provider_id in self._stored_providers:
            return {"type": "api_key", "source": "stored credential"}
        config = self._config.get_provider_override(provider_id)
        raw_key = _configured_api_key(config, extension)
        if raw_key is not None:
            if is_command_config_value(raw_key):
                return {"type": "api_key", "source": "configured API key"}
            if not get_config_value_env_var_names(raw_key):
                return {"type": "api_key", "source": "configured API key"}
            from .resolve_config_value import is_config_value_configured

            if is_config_value_configured(raw_key):
                return {"type": "api_key", "source": "configured API key"}
            return None
        base = self._native_extension_providers.get(provider_id) or self._builtins.get(provider_id)
        if base is not None and base.auth is None:
            return {"type": "api_key", "source": "no auth required"}
        return None

    def _recompose_provider(self, provider_id: str) -> None:
        """重组合单个 provider（失败时回退内置）。"""
        base = self._native_extension_providers.get(provider_id) or self._builtins.get(provider_id)
        config = self._config.get_provider_override(provider_id)
        extension = self._extension_providers.get(provider_id)
        if base is None and config is None and extension is None:
            self._models.remove_provider(provider_id)
            self._composition_errors.pop(provider_id, None)
            return
        if base is not None and config is None and extension is None:
            self._models.add_provider(base)
            self._composition_errors.pop(provider_id, None)
            return
        try:
            self._models.add_provider(self.compose_model_provider(provider_id))
            self._composition_errors.pop(provider_id, None)
        except Exception as exc:
            self._composition_errors[provider_id] = str(exc)
            if base is not None:
                self._models.add_provider(base)
            else:
                self._models.remove_provider(provider_id)

    def rebuild_providers(self) -> None:
        """清空并重建全部组合 provider。"""
        for provider in list(self._models.get_providers()):
            self._models.remove_provider(provider.id)
        self._composition_errors.clear()
        for provider_id in self._provider_ids():
            self._recompose_provider(provider_id)
        self._update_model_snapshot()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._models.get_provider(provider_id)

    def get_providers(self) -> list[Provider]:
        return self._models.get_providers()

    # ------------------------------------------------------------------
    # 模型查找
    # ------------------------------------------------------------------

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        return self._models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self._models.get_model(provider_id, model_id)

    # ------------------------------------------------------------------
    # 认证
    # ------------------------------------------------------------------

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        provider = self._models.get_provider(provider_id)
        if provider is None:
            return None
        return await self._check_provider_auth(provider)

    async def _check_provider_auth(self, provider: Provider) -> AuthCheck | None:
        provider_id = provider.id
        credential = await self._credentials.read(provider_id)
        if credential is not None:
            return {"type": credential_type(credential) or "api_key", "source": "stored credential"}
        auth = getattr(provider, "auth", None)
        if auth is None:
            # 本地/无认证 provider（如 faux、ollama）。
            return {"type": "api_key", "source": "no auth required"}
        resolved = auth.resolve(None)
        if resolved is not None:
            return {"type": "api_key", "source": resolved.source}
        return None

    def _resolve_configured_headers(
        self,
        model: Model | None,
        env: dict[str, str],
        raw_headers: dict[str, str] | None,
        auth_header: bool,
        api_key: str | None,
    ) -> dict[str, str | None] | None:
        merged: dict[str, str] = {}
        if raw_headers:
            resolved = resolve_headers_or_throw(
                raw_headers, f'provider "{model.provider if model else "?"}"', env
            )
            if resolved:
                merged.update(resolved)
        if model is not None and model.headers:
            resolved = resolve_headers_or_throw(
                model.headers, f'model "{model.provider}/{model.id}"', env
            )
            if resolved:
                merged.update(resolved)
        if auth_header:
            if api_key is None:
                raise ModelsError("auth", "authHeader requires a resolved API key")
            merged["Authorization"] = f"Bearer {api_key}"
        return cast(dict[str, str | None] | None, merged or None)

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: ModelRuntimeAuthOverrides | None = None,
    ) -> AuthResult | None:
        """解析 provider/model 认证（含配置 headers 合并）。"""
        overrides = overrides or {}
        if isinstance(provider_or_model, str):
            provider = self._models.get_provider(provider_or_model)
            if provider is None:
                return None
            return await self._resolve_auth(provider, None, overrides)
        model = provider_or_model
        provider = self._models.get_provider(model.provider)
        if provider is None:
            return None
        result = await self._resolve_auth(provider, model, overrides)
        if result is None:
            return None
        if model.headers:
            headers = {**(result.auth.get("headers") or {}), **model.headers}
            result.auth = {**result.auth, "headers": headers}
        return result

    async def _resolve_auth(
        self,
        provider: Provider,
        model: Model | None,
        overrides: ModelRuntimeAuthOverrides,
    ) -> AuthResult | None:
        provider_id = provider.id
        config = self._config.get_provider_override(provider_id)
        extension = self._extension_providers.get(provider_id)
        auth = getattr(provider, "auth", None)
        env = dict(overrides.get("env") or {})

        raw_key = _configured_api_key(config, extension)
        raw_headers = _configured_headers(config, extension)
        auth_header = bool(
            extension.get("auth_header")
            if extension is not None and extension.get("auth_header") is not None
            else (config.auth_header if config is not None else None)
        )

        explicit_key = overrides.get("api_key")
        if explicit_key:
            headers = self._resolve_configured_headers(
                model, env, raw_headers, auth_header, explicit_key
            )
            return AuthResult(
                auth=cast(ModelAuth, {"api_key": explicit_key, "headers": headers}),
                env=env or None,
                source="runtime API key",
            )

        credential = await self._credentials.read(provider_id)
        if credential is not None:
            ctype = credential_type(credential)
            if ctype == "oauth":
                oauth = getattr(auth, "oauth", None)
                if oauth is None:
                    return None
                # OAuth 流程（未来 provider 接入后启用）。
                from pi_ai.auth.resolve import resolve_stored_oauth

                return await resolve_stored_oauth(
                    self._credentials,
                    provider_id,
                    oauth,
                    credential,
                    overrides.get("min_oauth_validity_ms"),
                )
            key = _credential_key(credential)
            if key:
                headers = self._resolve_configured_headers(
                    model, env, raw_headers, auth_header, key
                )
                return AuthResult(
                    auth=cast(ModelAuth, {"api_key": key, "headers": headers}),
                    env=env or None,
                    source="stored credential",
                )
            return None

        if raw_key is not None:
            try:
                key = resolve_config_value_or_throw(
                    raw_key, f'API key for provider "{provider_id}"', env
                )
            except ValueError:
                return None
            headers = self._resolve_configured_headers(model, env, raw_headers, auth_header, key)
            return AuthResult(
                auth=cast(ModelAuth, {"api_key": key, "headers": headers}),
                env=env or None,
                source="configured API key",
            )

        if auth is not None and hasattr(auth, "resolve"):
            resolved = auth.resolve(None)
            if resolved is not None:
                headers = self._resolve_configured_headers(
                    model, env, raw_headers, auth_header, resolved.api_key
                )
                return AuthResult(
                    auth=cast(ModelAuth, {"api_key": resolved.api_key, "headers": headers}),
                    env=env or None,
                    source=resolved.source,
                )
            return None

        if auth is None:
            headers = self._resolve_configured_headers(model, env, raw_headers, auth_header, None)
            return AuthResult(
                auth=cast(ModelAuth, {"headers": headers}),
                source="no auth required",
            )
        return None

    # ------------------------------------------------------------------
    # 可用性快照
    # ------------------------------------------------------------------

    async def _run_availability_refresh(self) -> None:
        providers = self._models.get_providers()
        checks = await asyncio.gather(
            *(self._check_provider_auth(provider) for provider in providers)
        )
        credentials = await self._credentials.list()
        self._auth_checks = {
            provider.id: check for provider, check in zip(providers, checks, strict=True)
        }
        self._configured_providers = {
            provider_id for provider_id, check in self._auth_checks.items() if check is not None
        }
        self._stored_providers = {info["provider_id"] for info in credentials}
        all_models = self._models.get_models()
        self._available = [
            model for model in all_models if model.provider in self._configured_providers
        ]
        self._availability_error = None

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        if self._availability_refresh is not None:
            await self._availability_refresh
        else:
            await self._run_availability_refresh()
        if provider_id is not None:
            return [model for model in self._available if model.provider == provider_id]
        return list(self._available)

    def get_available_snapshot(self) -> list[Model]:
        return list(self._available)

    def has_configured_auth(self, provider_id: str) -> bool:
        return provider_id in self._configured_providers

    def is_using_oauth(self, provider_id: str) -> bool:
        check = self._auth_checks.get(provider_id)
        return check is not None and check.get("type") == "oauth"

    def get_provider_auth_status(self, provider_id: str) -> AuthStatus:
        if self._credentials.has_runtime_api_key(provider_id):
            return {"configured": True, "source": "runtime"}
        if provider_id in self._stored_providers:
            return {"configured": True, "source": "stored"}
        check = self._auth_checks.get(provider_id)
        if check is not None:
            return {"configured": True, "source": check.get("source")}
        return {"configured": False, "source": None}

    def list_credentials(self):
        return self._credentials.list()

    async def set_runtime_api_key(
        self, provider_id: str, api_key: str, refresh_options: ModelsRefreshOptions | None = None
    ) -> None:
        self._credentials.set_runtime_api_key(provider_id, api_key)
        self._auth_checks[provider_id] = {"type": "api_key", "source": "runtime API key"}
        self._configured_providers.add(provider_id)
        self._stored_providers.add(provider_id)
        self._available = [
            model
            for model in self._models.get_models()
            if model.provider in self._configured_providers
        ]
        await self.refresh(refresh_options or ModelsRefreshOptions(allow_network=False))

    async def remove_runtime_api_key(self, provider_id: str) -> None:
        self._credentials.remove_runtime_api_key(provider_id)
        await self.refresh(ModelsRefreshOptions(allow_network=False))

    async def logout(self, provider_id: str) -> None:
        await self._credentials.delete(provider_id)
        self._recompose_provider(provider_id)
        await self.refresh(ModelsRefreshOptions(allow_network=False))

    # ------------------------------------------------------------------
    # 动态刷新
    # ------------------------------------------------------------------

    def _update_model_snapshot(self) -> None:
        all_models = self._models.get_models()
        self._available = [
            model for model in all_models if model.provider in self._configured_providers
        ]

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        if self._models_path:
            self._config = await ModelConfig.load(self._models_path)
        self.rebuild_providers()
        result = await self._models.refresh(options)
        # Models.refresh 更新的是底层 provider 的动态模型；
        # 重新组合一次让 composed provider 的模型快照跟随刷新结果。
        self.rebuild_providers()
        self._update_model_snapshot()
        try:
            await self._run_availability_refresh()
        except Exception as exc:
            self._availability_error = str(exc)
        return result

    # ------------------------------------------------------------------
    # 状态与错误
    # ------------------------------------------------------------------

    def get_error(self) -> str | None:
        errors: list[str] = []
        config_error = self._config.get_error()
        if config_error:
            errors.append(config_error)
        for provider_id, error in self._composition_errors.items():
            errors.append(f'Provider "{provider_id}": {error}')
        if self._availability_error:
            errors.append(f"Availability refresh: {self._availability_error}")
        return "\n\n".join(errors) if errors else None

    def get_registered_provider_config(self, provider_id: str) -> ProviderConfigInput | None:
        return self._extension_providers.get(provider_id)

    def get_registered_provider_ids(self) -> list[str]:
        return list(set(self._extension_providers) | set(self._native_extension_providers))

    def get_registered_native_provider(self, provider_id: str) -> Provider | None:
        return self._native_extension_providers.get(provider_id)

    def get_compatibility_request_config(self, model: Model) -> CompatibilityRequestConfig:
        config = self._config.get_provider_override(model.provider)
        extension = self._extension_providers.get(model.provider)
        raw_headers = _configured_headers(config, extension)
        merged = dict(raw_headers or {})
        if model.headers:
            merged.update(model.headers)
        return {
            "headers": cast(dict[str, str], merged or None),
            "auth_header": bool(
                extension.get("auth_header")
                if extension is not None and extension.get("auth_header") is not None
                else (config.auth_header if config is not None else None)
            ),
        }

    # ------------------------------------------------------------------
    # 流式接口（pi_ai Models 兼容）
    # ------------------------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return await self._models.stream(model, context, options)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        return await self._models.complete(model, context, options)

    def supports_deferred(self, model: Model) -> bool:
        """模型所属 Provider 是否支持挂起响应。"""
        return self._models.supports_deferred(model)

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: dict[str, Any] | None = None,
    ) -> AssistantMessage:
        """抓取挂起响应（委托 Models）。"""
        return await self._models.fetch_deferred(model, handle, options)

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: dict[str, Any] | None = None,
    ) -> None:
        """取消挂起响应（委托 Models）。"""
        await self._models.cancel_deferred(model, handle, options)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.refresh(ModelsRefreshOptions(allow_network=False)))


def _validate_extension_provider(
    provider_id: str,
    base: Provider | None,
    models_config: ProviderOverride | None,
    extension: ProviderConfigInput,
) -> None:
    if extension.get("stream_simple") is not None and not extension.get("api"):
        raise ValueError(
            f'Provider {provider_id}: "api" is required when registering streamSimple.'
        )
    # 结构校验：组合模型失败即抛错。
    _compose_models(provider_id, base, models_config, extension)


__all__ = [
    "ModelRuntime",
    "AuthCheck",
    "AuthStatus",
    "ModelRuntimeAuthOverrides",
    "CompatibilityRequestConfig",
    "ProviderConfigInput",
    "ComposedApiKeyAuth",
]
