"""凭证存储（对齐 TS auth/credential-store.ts）。

- InMemoryCredentialStore：按 provider 串行化的内存实现（含 modify/list）；
- FileCredentialStore：auth.json 文件持久化（原子写）。
"""

import asyncio
import json

from pathlib import Path
from typing import Any

from . import ApiKeyCredential
from .types import Credential, CredentialInfo, credential_type


def _to_raw(credential: Credential) -> dict[str, Any]:
    """凭证 → JSON 可序列化字典。"""
    if isinstance(credential, dict):
        return dict(credential)
    return {"type": credential.type, "key": credential.key}


def _from_raw(raw: dict[str, Any]) -> Credential:
    """JSON 字典 → 凭证（api_key 恢复为 dataclass，oauth 保持 dict）。"""
    if raw.get("type") == "oauth":
        return raw
    return ApiKeyCredential(type="api_key", key=raw.get("key"))


class InMemoryCredentialStore:
    """内存凭证存储；modify/delete 按 provider 串行化。"""

    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock

    async def read(self, provider_id: str) -> Credential | None:
        return self._credentials.get(provider_id)

    async def list(self) -> list[CredentialInfo]:
        return [
            {
                "provider_id": provider_id,
                "type": credential_type(credential) or "",
            }
            for provider_id, credential in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn,
    ) -> Credential | None:
        async with self._lock(provider_id):
            current = self._credentials.get(provider_id)
            next_value = await fn(current)
            if next_value is not None:
                self._credentials[provider_id] = next_value
            return next_value if next_value is not None else current

    async def delete(self, provider_id: str) -> None:
        async with self._lock(provider_id):
            self._credentials.pop(provider_id, None)

    # 兼容旧 API（Provider/测试使用）。
    async def write(self, provider_id: str, credential: Credential) -> None:
        async def _set(_current):
            return credential

        await self.modify(provider_id, _set)


class FileCredentialStore:
    """auth.json 文件持久化（进程内按 provider 串行化 + 原子写）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock

    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 凭证文件仅限本人读写（对齐 TS chmodSync(0o600)）。
        tmp.chmod(0o600)
        tmp.replace(self._path)
        self._path.chmod(0o600)

    async def read(self, provider_id: str) -> Credential | None:
        raw = self._load().get(provider_id)
        return _from_raw(raw) if isinstance(raw, dict) else None

    async def list(self) -> list[CredentialInfo]:
        return [
            {
                "provider_id": provider_id,
                "type": raw.get("type", "") if isinstance(raw, dict) else "",
            }
            for provider_id, raw in self._load().items()
        ]

    async def modify(self, provider_id: str, fn) -> Credential | None:
        async with self._lock(provider_id):
            data = self._load()
            raw = data.get(provider_id)
            current = _from_raw(raw) if isinstance(raw, dict) else None
            next_value = await fn(current)
            if next_value is not None:
                data[provider_id] = _to_raw(next_value)
                self._save(data)
            return next_value if next_value is not None else current

    async def delete(self, provider_id: str) -> None:
        async with self._lock(provider_id):
            data = self._load()
            if provider_id in data:
                del data[provider_id]
                self._save(data)

    # 兼容旧 API。
    async def write(self, provider_id: str, credential: Credential) -> None:
        async def _set(_current):
            return credential

        await self.modify(provider_id, _set)


__all__ = ["InMemoryCredentialStore", "FileCredentialStore"]
