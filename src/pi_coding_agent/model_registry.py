"""ModelRegistry — ModelRuntime 的同步兼容外观（对齐 TS core/model-registry.ts）。

coding-agent 内部直接使用 ModelRuntime；本外观供扩展系统使用。
"""

from __future__ import annotations

from typing import TypedDict

from pi_ai import Model, Provider

from .model_runtime import ModelRuntime, ProviderConfigInput


class ResolvedRequestAuth(TypedDict):
    ok: bool
    api_key: str | None
    headers: dict[str, str] | None
    env: dict[str, str] | None
    error: str | None


class ModelRegistry:
    """同步兼容外观：扩展 / 旧代码访问 ModelRuntime 的统一入口。"""

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    async def refresh(self) -> None:
        """重载 models.json 并刷新（同步读取前先 await）。"""
        await self._runtime.refresh()

    def get_error(self) -> str | None:
        return self._runtime.get_error()

    def get_all(self) -> list[Model]:
        return list(self._runtime.get_models())

    def get_available(self) -> list[Model]:
        return self._runtime.get_available_snapshot()

    def find(self, provider: str, model_id: str) -> Model | None:
        return self._runtime.get_model(provider, model_id)

    def find_by_id(self, model_id: str) -> Model | None:
        for model in self._runtime.get_models():
            if model.id == model_id:
                return model
        return None

    def has_configured_auth(self, model: Model) -> bool:
        return self._runtime.has_configured_auth(model.provider)

    async def get_api_key_and_headers(self, model: Model) -> ResolvedRequestAuth:
        """解析模型认证（同步外观的异步方法）。"""
        try:
            resolution = await self._runtime.get_auth(model)
        except Exception as exc:
            return {
                "ok": False,
                "api_key": None,
                "headers": None,
                "env": None,
                "error": str(exc),
            }
        if resolution is None:
            compatibility = self._runtime.get_compatibility_request_config(model)
            if compatibility.get("auth_header"):
                return {
                    "ok": False,
                    "api_key": None,
                    "headers": None,
                    "env": None,
                    "error": f'No API key found for "{model.provider}"',
                }
            return {
                "ok": True,
                "api_key": None,
                "headers": compatibility.get("headers"),
                "env": None,
                "error": None,
            }
        return {
            "ok": True,
            "api_key": resolution.auth.get("api_key"),
            "headers": resolution.auth.get("headers"),
            "env": resolution.env,
            "error": None,
        }

    def get_provider_auth_status(self, provider: str):
        return self._runtime.get_provider_auth_status(provider)

    def get_provider(self, provider: str) -> Provider | None:
        return self._runtime.get_provider(provider)

    def get_provider_display_name(self, provider: str) -> str:
        found = self._runtime.get_provider(provider)
        return found.name if found is not None else provider

    async def get_provider_auth(self, provider: str):
        return await self._runtime.get_auth(provider)

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        try:
            resolution = await self._runtime.get_auth(provider)
        except Exception:
            return None
        return resolution.auth.get("api_key") if resolution is not None else None

    def is_using_oauth(self, model: Model) -> bool:
        return self._runtime.is_using_oauth(model.provider)

    def register_provider(
        self, provider_or_name: Provider | str, config: ProviderConfigInput | None = None
    ):
        if isinstance(provider_or_name, str):
            if not config:
                raise ValueError("Provider config is required when registering by name")
            self._runtime.register_provider(provider_or_name, config)
            return
        self._runtime.register_native_provider(provider_or_name)

    def unregister_provider(self, provider_name: str) -> None:
        self._runtime.unregister_provider(provider_name)

    def get_registered_provider_config(self, provider_name: str) -> ProviderConfigInput | None:
        return self._runtime.get_registered_provider_config(provider_name)

    def get_registered_native_provider(self, provider_name: str) -> Provider | None:
        return self._runtime.get_registered_native_provider(provider_name)

    def get_registered_provider_ids(self) -> list[str]:
        return self._runtime.get_registered_provider_ids()


__all__ = ["ModelRegistry", "ResolvedRequestAuth"]
