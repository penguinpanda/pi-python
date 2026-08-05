"""图片 Provider 集合（对齐 TS images-models.ts）。

- ImagesProvider：图片侧 provider（与聊天 Provider 平行）；
- ImagesModels：注册 / 查询 / refresh / getAuth / generateImages；
- generateImages 永不 reject：失败返回 stopReason="error" 的 AssistantImages。
"""

import asyncio

from typing import Any, Awaitable, Callable, Protocol, cast

from .types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    now_ms,
)
from .auth.context import AuthContext, default_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import ModelsError, resolve_provider_auth
from .auth.types import CredentialStore

ImagesFunction = Callable[
    [ImagesModel, ImagesContext, ImagesOptions | None],
    Awaitable[AssistantImages],
]


class ImagesProvider(Protocol):
    """图片生成 provider 接口（与聊天 Provider 平行）。"""

    id: str
    name: str
    auth: Any  # EnvApiKeyAuth | 带 oauth 的对象 | None

    def get_models(self) -> list[ImagesModel]: ...

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages: ...

    # 动态 provider 可选
    refresh_models: Any


class ImagesModels:
    """图片生成 provider 集合。"""

    def __init__(
        self,
        credentials: CredentialStore | None = None,
        auth_context: AuthContext | None = None,
    ) -> None:
        self._providers: dict[str, ImagesProvider] = {}
        self._credentials = credentials or InMemoryCredentialStore()
        self._auth_context = auth_context or default_auth_context()

    # 注册管理

    def set_provider(self, provider: ImagesProvider) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_providers(self) -> list[ImagesProvider]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> ImagesProvider | None:
        return self._providers.get(provider_id)

    # 模型查询

    def get_models(self, provider: str | None = None) -> list[ImagesModel]:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return []
            try:
                return entry.get_models()
            except Exception:
                return []
        models: list[ImagesModel] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:
                pass
        return models

    def get_model(self, provider: str, model_id: str) -> ImagesModel | None:
        return next((m for m in self.get_models(provider) if m.id == model_id), None)

    # 动态刷新

    async def refresh(self, provider: str | None = None) -> None:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None or not callable(getattr(entry, "refresh_models", None)):
                return
            try:
                await entry.refresh_models()
            except Exception as exc:
                if isinstance(exc, ModelsError):
                    raise
                raise ModelsError(
                    "model_source",
                    f"Model refresh failed for {provider}",
                    cause=exc,
                ) from exc
            return
        # 全部并发 best-effort，不 reject。
        await asyncio.gather(
            *(
                entry.refresh_models()
                for entry in self._providers.values()
                if callable(getattr(entry, "refresh_models", None))
            ),
            return_exceptions=True,
        )

    # 认证

    async def get_auth(
        self,
        provider_or_model: str | ImagesModel,
        overrides: dict[str, Any] | None = None,
    ):
        provider_id = (
            provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        )
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        return await resolve_provider_auth(
            provider, self._credentials, self._auth_context, overrides
        )

    # 生成

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        try:
            provider = self._providers.get(model.provider)
            if provider is None:
                raise ModelsError("provider", f"Unknown provider: {model.provider}")

            options = options or {}
            resolution = await self.get_auth(
                model,
                {
                    "api_key": options.get("api_key"),
                    "env": options.get("env"),
                },
            )
            auth = resolution.auth if resolution is not None else None

            api_key = options.get("api_key") or (auth.get("api_key") if auth else None)
            headers = options.get("headers")
            if auth and auth.get("headers"):
                merged = dict(auth["headers"])
                if headers:
                    merged.update({k: v for k, v in headers.items() if v is not None})
                headers = merged
            request_options = cast(ImagesOptions, dict(options))
            if api_key:
                request_options["api_key"] = api_key
            if headers:
                request_options["headers"] = headers

            return await provider.generate_images(model, context, request_options)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "api": model.api,
                "provider": model.provider,
                "model": model.id,
                "output": [],
                "stop_reason": "error",
                "error_message": str(exc),
                "timestamp": now_ms(),
            }


def create_images_models(
    credentials: CredentialStore | None = None,
    auth_context: AuthContext | None = None,
) -> ImagesModels:
    return ImagesModels(credentials=credentials, auth_context=auth_context)


class _SimpleImagesProvider:
    """由 create_images_provider 构建的 ImagesProvider 实现。"""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        auth: Any,
        models: list[ImagesModel],
        api: ImagesFunction | Any,
    ) -> None:
        self.id = id
        self.name = name
        self.auth = auth
        self._models = list(models)
        self._api = api
        self._inflight: asyncio.Task | None = None
        self.refresh_models: Any = None

    def get_models(self) -> list[ImagesModel]:
        return list(self._models)

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        if callable(self._api):
            return await self._api(model, context, options)
        return await self._api.generate_images(model, context, options)

    def _set_refresh(self, fetch) -> None:
        async def _do_refresh() -> None:
            if self._inflight is not None:
                await self._inflight
                return

            async def _impl() -> None:
                try:
                    refreshed = await fetch()
                    self._models[:] = refreshed
                finally:
                    self._inflight = None

            task = asyncio.create_task(_impl())
            self._inflight = task
            try:
                await task
            finally:
                self._inflight = None

        self.refresh_models = _do_refresh


def create_images_provider(
    *,
    id: str,
    name: str | None = None,
    auth: Any,
    models: list[ImagesModel],
    api: ImagesFunction | Any,
    refresh_models: Callable[[], Awaitable[list[ImagesModel]]] | None = None,
) -> ImagesProvider:
    """从部件构建图片 provider。"""
    provider = _SimpleImagesProvider(
        id=id,
        name=name or id,
        auth=auth,
        models=models,
        api=api,
    )
    if refresh_models is not None:
        provider._set_refresh(refresh_models)
    return provider  # type: ignore[return-value]


__all__ = [
    "ImagesProvider",
    "ImagesModels",
    "create_images_models",
    "create_images_provider",
]
