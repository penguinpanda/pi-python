"""凭证存储（对齐 TS auth/credential-store.ts）。

- InMemoryCredentialStore：按 provider 串行化的内存实现（含 modify/list）；
- FileCredentialStore：auth.json 文件持久化（原子写）。
"""

import asyncio
import json

from filelock import FileLock
from pathlib import Path
from typing import Any

from . import ApiKeyCredential
from .types import Credential, CredentialInfo, credential_type
from ..utils.atomic_write import atomic_write_json


class CredentialStoreCorruptError(Exception):
    """凭证文件损坏（JSON 解析失败）；原文件已备份，拒绝静默覆盖。"""


def _backup_corrupt_file(path: Path) -> None:
    """把损坏的凭证文件移到 .corrupt 备份，避免后续写入吞掉原始数据。"""
    backup = path.with_name(path.name + ".corrupt")
    try:
        path.replace(backup)
    except OSError:
        pass


def _load(path: Path) -> dict[str, Any]:
    """读取凭证文件；损坏时备份并抛 CredentialStoreCorruptError。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        _backup_corrupt_file(path)
        raise CredentialStoreCorruptError(f"Credential file {path} is corrupted") from exc


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
    """auth.json 文件持久化（进程内全局串行化 + 跨进程 filelock + 原子写）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # 进程内全局锁：filelock 在同一线程内可重入，无法防止并发协程
        # 各自持有旧文件快照后整文件写回（跨 provider 丢失更新）。
        # 所有读-改-写全程在此锁内串行化；filelock 仅负责跨进程互斥。
        self._global_lock = asyncio.Lock()
        self._file_lock = FileLock(str(self._path) + ".lock", timeout=30)

    async def read(self, provider_id: str) -> Credential | None:
        try:
            raw = _load(self._path).get(provider_id)
        except CredentialStoreCorruptError:
            return None
        return _from_raw(raw) if isinstance(raw, dict) else None

    async def list(self) -> list[CredentialInfo]:
        try:
            data = _load(self._path)
        except CredentialStoreCorruptError:
            return []
        return [
            {
                "provider_id": provider_id,
                "type": raw.get("type", "") if isinstance(raw, dict) else "",
            }
            for provider_id, raw in data.items()
        ]

    async def modify(self, provider_id: str, fn) -> Credential | None:
        async with self._global_lock:
            with self._file_lock:
                data = _load(self._path)
                raw = data.get(provider_id)
                current = _from_raw(raw) if isinstance(raw, dict) else None
                next_value = await fn(current)
                if next_value is not None:
                    data[provider_id] = _to_raw(next_value)
                    atomic_write_json(self._path, data)
                return next_value if next_value is not None else current

    async def delete(self, provider_id: str) -> None:
        async with self._global_lock:
            with self._file_lock:
                data = _load(self._path)
                if provider_id in data:
                    del data[provider_id]
                    atomic_write_json(self._path, data)

    # 兼容旧 API。
    async def write(self, provider_id: str, credential: Credential) -> None:
        async def _set(_current):
            return credential

        await self.modify(provider_id, _set)


__all__ = ["InMemoryCredentialStore", "FileCredentialStore"]
