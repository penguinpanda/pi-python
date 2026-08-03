"""运行时 API Key 覆盖层（对齐 TS core/runtime-credentials.ts）。

非持久化的 CredentialStore 包装：`setRuntimeApiKey` 设置的 key
优先于底层存储，且不写入 auth.json。
"""

from __future__ import annotations

from pi_ai.auth import ApiKeyCredential
from pi_ai.auth.types import Credential, CredentialInfo, CredentialStore


class RuntimeCredentials:
    """异步 CredentialStore 覆盖层（进程内 runtime API key）。"""

    def __init__(self, store: CredentialStore) -> None:
        self._store = store
        self._overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self._overrides[provider_id] = api_key

    def remove_runtime_api_key(self, provider_id: str) -> None:
        self._overrides.pop(provider_id, None)

    def has_runtime_api_key(self, provider_id: str) -> bool:
        return provider_id in self._overrides

    async def read(self, provider_id: str) -> Credential | None:
        override = self._overrides.get(provider_id)
        if override is not None:
            return ApiKeyCredential(type="api_key", key=override)
        return await self._store.read(provider_id)

    async def list(self) -> list[CredentialInfo]:
        entries = {
            info["provider_id"]: info for info in await self._store.list()
        }
        for provider_id in self._overrides:
            entries[provider_id] = {"provider_id": provider_id, "type": "api_key"}
        return list(entries.values())

    async def modify(self, provider_id: str, fn) -> Credential | None:
        return await self._store.modify(provider_id, fn)

    async def delete(self, provider_id: str) -> None:
        self._overrides.pop(provider_id, None)
        await self._store.delete(provider_id)


__all__ = ["RuntimeCredentials"]
