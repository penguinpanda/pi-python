"""
模型注册表（Models）

=========================================================
模块职责
=========================================================

Models 是整个 SDK 的统一入口（Facade）。

它负责：

    ① Provider 注册

            add_provider()

    ② Model 查询

            get_model()

            get_models()

    ③ 请求调度

            stream()

            complete()

    ④ API Key 管理

            set_api_key()

Models 本身并不会直接调用 OpenAI、DeepSeek 等模型接口。

真正发送 HTTP 请求的是 Provider。

因此：

            Models
               │
               ▼
          找到 Provider
               │
               ▼
      Provider.stream(...)
               │
               ▼
         OpenAI / DeepSeek


=========================================================
整体关系
=========================================================

                    Models
                       │
       ┌───────────────┴───────────────┐
       │                               │
 Provider(OpenAI)              Provider(DeepSeek)
       │                               │
   GPT-4o                       DeepSeek Chat
   GPT-5                        DeepSeek Reasoner

Models 相当于整个 SDK 的调度中心，
负责根据 Model.provider 找到对应 Provider。
"""

import asyncio
import os

from dataclasses import dataclass, field
from typing import Any

from ..utils._event_stream import AssistantMessageEventStream
from ..types import (
    AssistantMessage,
    Context,
    DeferredHandle,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from ..auth import InMemoryCredentialStore
from ..auth.context import AuthContext, default_auth_context
from ..auth.resolve import ModelsError, resolve_provider_auth, resolve_stored_oauth
from ..auth.types import AuthInteraction, AuthResult, Credential, credential_type
from pi_telemetry import SpanOptions
from ..provider import Provider, RefreshModelsContext
from .models_store import (
    InMemoryModelsStore,
    ModelsStore,
    provider_models_store,
)


@dataclass(slots=True)
class ModelsRefreshOptions:
    """Models.refresh 选项。"""

    # False 表示离线/仅缓存初始化（不访问网络）。
    allow_network: bool = True
    # 绕过 provider 新鲜度检查立即抓取。
    force: bool = False
    # 可选中止信号（asyncio.Event）。
    signal: asyncio.Event | None = None


@dataclass(slots=True)
class ModelsRefreshResult:
    """Models.refresh 结果：aborted + 每 provider 的错误（不抛给调用方）。"""

    aborted: bool = False
    errors: dict[str, Exception] = field(default_factory=dict)


def _resolve_env_credential(provider: Provider) -> Credential | None:
    """环境变量认证回退（对齐 TS resolveRefreshCredential）。

    refresh 只读 CredentialStore 时，仅用环境变量 API Key 的动态 provider
    会拿到空凭证，抓取返回空列表并覆写缓存。此处按 provider.auth 的
    env_vars 解析 ambient key（与 _DefaultAuthContext.env 语义一致）。
    """
    auth = getattr(provider, "auth", None)
    env_vars = getattr(auth, "env_vars", None)
    if not env_vars:
        return None
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            return {"type": "api_key", "key": value}
    return None


class Models:
    """
    模型管理器（Model Registry）。

    整个 SDK 的统一入口。

    主要职责：

    1. 注册 Provider
    2. 查询 Model
    3. 调度请求
    4. 管理 API Key

    Models 本身不实现任何模型接口。

    所有网络请求都会转发给对应 Provider。
    """

    def __init__(
        self,
        *,
        models_store: ModelsStore | None = None,
        credentials=None,
        auth_context: AuthContext | None = None,
    ) -> None:

        # 已注册的 Provider
        #
        # key:
        #     provider.id
        #
        # value:
        #     Provider
        #
        # 例如：
        #
        # {
        #     "openai": OpenAIProvider(...),
        #     "deepseek": DeepSeekProvider(...)
        # }
        self._providers: dict[str, Provider] = {}

        # 默认使用内存 Credential Store。
        #
        # Provider 获取 API Key 时，
        # 会统一从这里读取。
        #
        # credentials 可注入外部实现（例如 coding-agent 的 AuthStorage），
        # 满足 CredentialStore 接口（read/list/modify/delete）即可。
        self._credentials = credentials or InMemoryCredentialStore()
        self._auth_context = auth_context or default_auth_context()

        # 模型目录持久化（ModelsStore）。
        self._models_store = models_store or InMemoryModelsStore()

    # 模型提供者 管理

    def add_provider(self, provider: Provider) -> None:
        """
        注册一个 Provider。

        如果 Provider ID 已存在，

        则新的 Provider 会覆盖旧 Provider。

        例如：

            add_provider(OpenAIProvider())

        之后即可：

            models.stream(...)
        """

        provider._credential_store = self._credentials
        self._providers[provider.id] = provider

    def get_provider(self, provider_id: str) -> Provider | None:
        """
        根据 Provider ID 获取 Provider。

        例如：

            get_provider("openai")
        """

        return self._providers.get(provider_id)

    def remove_provider(self, provider_id: str) -> None:
        """
        移除一个 Provider。

        如果 Provider 不存在，

        不会抛异常。
        """

        self._providers.pop(provider_id, None)

    def get_providers(self) -> list[Provider]:
        """
        返回所有已注册 Provider。
        """

        return list(self._providers.values())

    # 模型查找

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        """
        获取模型列表。

        provider_id=None

        返回：

            所有 Provider 的模型。

        provider_id="openai"

        返回：

            OpenAI 的模型。
        """

        if provider_id is not None:
            provider = self._providers.get(provider_id)
            return provider.get_models() if provider else []

        result: list[Model] = []

        for provider in self._providers.values():
            result.extend(provider.get_models())
        return result

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        """
        根据：

        Provider ID + Model ID

        查找一个模型。

        例如：

            get_model(
                "openai",
                "gpt-4o"
            )
        """

        provider = self._providers.get(provider_id)
        if not provider:
            return None

        for model in provider.get_models():
            if model.id == model_id:
                return model

        return None

    def get_model_by_id(self, model_id: str) -> Model | None:
        """
        跨所有 Provider 按模型 ID 查找。

        当不关心 Provider 归属时，
        直接按 ID 全局查找。

        例如：

            get_model_by_id("deepseek-chat")

        返回第一个匹配的模型。
        """

        for provider in self._providers.values():
            for model in provider.get_models():
                if model.id == model_id:
                    return model

        return None

    # 动态模型刷新

    async def refresh(
        self,
        options: ModelsRefreshOptions | None = None,
    ) -> ModelsRefreshResult:
        """并发刷新所有支持 refreshModels 的 provider。

        对齐 TS Models.refresh：
        - 单个 provider 失败记录到 errors，不整体抛异常；
        - 失败后以 allow_network=False 重跑一次做缓存恢复（best-effort）；
        - 中止（signal set）不产生错误。
        """
        opts = options or ModelsRefreshOptions()
        result = ModelsRefreshResult()
        refreshable = [
            provider
            for provider in self._providers.values()
            if getattr(provider, "refresh_models", None) is not None
        ]

        async def _refresh_one(provider: Provider) -> None:
            refresh_models = provider.refresh_models
            assert refresh_models is not None
            if opts.signal is not None and opts.signal.is_set():
                result.aborted = True
                return
            store = provider_models_store(self._models_store, provider.id)
            credential = await self._credentials.read(provider.id)
            if credential is None:
                credential = _resolve_env_credential(provider)
            elif credential_type(credential) == "oauth":
                # 目录抓取前先刷新过期 OAuth token（与 Provider.stream 路径
                # 一致）：否则带过期 access token 请求 /v1/config 直接 401，
                # 目录停留在旧缓存。
                oauth = getattr(getattr(provider, "auth", None), "oauth", None)
                if oauth is not None:
                    try:
                        await resolve_stored_oauth(
                            self._credentials, provider.id, oauth, credential
                        )
                    except ModelsError:
                        pass  # 刷新失败不阻断;抓取阶段的错误由 result.errors 收集
                    refreshed_credential = await self._credentials.read(provider.id)
                    if refreshed_credential is not None:
                        credential = refreshed_credential
            context = RefreshModelsContext(
                credential=credential,
                store=store,
                allow_network=opts.allow_network,
                force=opts.force,
                signal=opts.signal,
            )
            try:
                await refresh_models(context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if opts.signal is not None and opts.signal.is_set():
                    result.aborted = True
                    return
                result.errors[provider.id] = exc
                # 失败后 best-effort 缓存恢复（离线重跑）。
                try:
                    await refresh_models(
                        RefreshModelsContext(
                            credential=credential,
                            store=store,
                            allow_network=False,
                            force=opts.force,
                            signal=opts.signal,
                        )
                    )
                except Exception:
                    pass

        await asyncio.gather(*(_refresh_one(p) for p in refreshable))
        return result

    # 请求调度

    async def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        """
        发起流式请求。

        工作流程：

                立即返回 EventStream

                后台异步 setup（认证解析 + Provider 调度）

                失败 → error 事件优雅降级（对齐 TS lazyStream）
        """

        from ..api.lazy import lazy_stream

        async def _setup():
            provider = self._require_provider(model.provider)
            opts = options or {}
            telemetry = opts.get("telemetry_context")
            if telemetry is None:
                return await provider.stream(model, context, options)
            return await telemetry.start_span(
                SpanOptions(
                    name="pi.ai.request",
                    attributes={
                        "pi.ai.provider": model.provider,
                        "pi.ai.model": model.id,
                        "pi.ai.api": model.api,
                    },
                ),
                lambda _span: provider.stream(model, context, options),
            )

        return lazy_stream(model, _setup)

    async def complete(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessage:
        """
        非流式调用。

        内部实际上仍然使用 stream()。

        区别仅在于：

        stream()

        返回：

            EventStream

        complete()

        等待整个 EventStream 完成，

        直接返回最终 AssistantMessage。
        """

        stream = await self.stream(model, context, options)
        return await stream.result()

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """发起 streamSimple 请求（对齐 TS Models.streamSimple）。"""

        from ..api.lazy import lazy_stream

        async def _setup():
            provider = self._require_provider(model.provider)
            opts = options or {}
            telemetry = opts.get("telemetry_context")
            if telemetry is None:
                return await provider.stream_simple(model, context, options)
            return await telemetry.start_span(
                SpanOptions(
                    name="pi.ai.request",
                    attributes={
                        "pi.ai.provider": model.provider,
                        "pi.ai.model": model.id,
                        "pi.ai.api": model.api,
                    },
                ),
                lambda _span: provider.stream_simple(model, context, options),
            )

        return lazy_stream(model, _setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        """completeSimple：等待整个流结束，返回最终 AssistantMessage。"""
        stream = await self.stream_simple(model, context, options)
        return await stream.result()

    def supports_deferred(self, model: Model) -> bool:
        """模型所属 Provider 是否支持挂起响应。"""
        provider = self.get_provider(model.provider)
        return provider is not None and provider._deferred_fn is not None

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: dict[str, Any] | None = None,
    ) -> AssistantMessage:
        """抓取挂起响应（对齐 TS Models.fetchDeferred）。"""
        provider = self._require_provider(model.provider)
        return await provider.fetch_deferred(model, handle, options)

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: dict[str, Any] | None = None,
    ) -> None:
        """取消挂起响应（对齐 TS Models.cancelDeferred）。"""
        provider = self._require_provider(model.provider)
        await provider.cancel_deferred(model, handle, options)

    # 凭证管理

    async def set_api_key(self, provider_id: str, api_key: str) -> None:
        """
        保存 Provider 的 API Key。

        保存以后：

        Provider 会优先使用这里保存的 Key，

        而不是环境变量。
        """

        from ..auth import ApiKeyCredential

        await self._credentials.write(provider_id, ApiKeyCredential(key=api_key))

    # 认证查询与管理（对齐 TS Models.getAuth/checkAuth/login/logout）

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: dict[str, Any] | None = None,
    ) -> AuthResult | None:
        """解析 provider/model 认证（API Key 或 OAuth，含刷新）。"""
        provider_id = (
            provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        )
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        result = await resolve_provider_auth(
            provider,
            self._credentials,
            self._auth_context,
            overrides,
        )
        if (
            result is not None
            and isinstance(provider_or_model, Model)
            and provider_or_model.headers
        ):
            headers = dict(result.auth.get("headers") or {})
            headers.update(provider_or_model.headers)
            result.auth = {**result.auth, "headers": headers}
        return result

    async def check_auth(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """检查 provider 是否已有可用的认证配置。"""
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        if getattr(provider, "auth", None) is None:
            return {"type": "api_key", "source": "no auth required"}
        credential = await self._credentials.read(provider_id)
        if credential is not None:
            ctype = credential_type(credential)
            if ctype == "oauth":
                oauth = getattr(provider.auth, "oauth", None)
                return {"type": "oauth", "source": "OAuth"} if oauth is not None else None
            if ctype == "api_key":
                return {"type": "api_key", "source": "stored credential"}
        resolver = getattr(provider.auth, "resolve_auth", None)
        if callable(resolver):
            custom = await resolver(
                self._credentials,
                self._auth_context,
                overrides or {},
            )
            if custom is not None:
                return {"type": "api_key", "source": "custom"}
        result = await resolve_provider_auth(
            provider,
            self._credentials,
            self._auth_context,
            overrides,
        )
        if result is None:
            return None
        return {"type": "api_key", "source": result.source or "env"}

    async def get_available(
        self,
        provider_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> list[Model]:
        """返回已配置认证的 provider 的模型列表。"""
        providers = (
            [self._providers[provider_id]]
            if provider_id is not None and provider_id in self._providers
            else list(self._providers.values())
        )
        available: list[Model] = []
        for provider in providers:
            if await self.check_auth(provider.id, overrides) is not None:
                available.extend(provider.get_models())
        available.sort(key=lambda model: (model.provider, model.id))
        return available

    async def login(
        self,
        provider_id: str,
        interaction: AuthInteraction,
    ) -> Credential:
        """运行 provider 的 OAuth 登录流程并持久化凭证。"""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("auth", f"Unknown provider: {provider_id}")
        auth = getattr(provider, "auth", None)
        oauth = getattr(auth, "oauth", None)
        login_method = getattr(auth, "login", None)
        if oauth is not None:
            credential = await oauth.login(interaction)
        elif callable(login_method):
            credential = await login_method(interaction)
        else:
            raise ModelsError("auth", f"Provider {provider_id} does not support login")
        await self._credentials.write(provider_id, credential)
        return credential

    async def logout(self, provider_id: str) -> None:
        """删除 provider 的存储凭证。"""
        await self._credentials.delete(provider_id)

    # 辅助函数

    def _require_provider(self, provider_id: str) -> Provider:
        """
        获取指定 Provider。

        如果 Provider 不存在，

        抛出 ValueError。

        这是一个内部辅助函数，

        避免在多个地方重复判断：

        if provider is None
        """

        provider = self._providers.get(provider_id)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_id}")
        return provider
