"""认证持久化（对齐 TS core/auth-storage.ts）。

AuthStorage 实现 pi_ai 的 CredentialStore 接口，提供：

- 文件后端：~/.pi/agent/auth.json，原子写 + filelock 并发保护；
- 内存后端：测试 / 进程内场景；
- 环境变量引用：读取 api_key 时解析 `$ENV_VAR` / `${ENV_VAR}` / `!command`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, TypedDict

from filelock import FileLock
from pi_ai.auth import ApiKeyCredential
from pi_ai.auth.types import Credential, CredentialInfo, credential_type

from ._config import get_agent_dir
from .resolve_config_value import resolve_config_value


class LockResult(TypedDict):
    result: Any
    next: str | None


class AuthStorageBackend(Protocol):
    """带锁的原始 JSON 内容存储（对齐 TS AuthStorageBackend）。"""

    def with_lock(self, fn: Any) -> Any: ...

    async def with_lock_async(self, fn: Any) -> Any: ...


def _lock_path(path: Path) -> str:
    return str(path) + ".lock"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    # auth.json 含 API key / oauth token，仅限本人读写（对齐 TS mode: 0o600 + chmodSync）。
    tmp.chmod(0o600)
    tmp.replace(path)
    path.chmod(0o600)


class FileAuthStorageBackend:
    """auth.json 文件后端（原子写 + 文件级锁）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else get_agent_dir() / "auth.json"

    def _ensure(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            _write_text(self._path, "{}")

    def with_lock(self, fn: Any) -> Any:
        self._ensure()
        with FileLock(_lock_path(self._path), timeout=30):
            current = _read_text(self._path)
            result = fn(current)
            next_value = result.get("next") if isinstance(result, dict) else None
            if next_value is not None:
                _write_text(self._path, next_value)
            return result.get("result") if isinstance(result, dict) else result

    async def with_lock_async(self, fn: Any) -> Any:
        self._ensure()
        lock = FileLock(_lock_path(self._path), timeout=30)
        # filelock 的锁是线程本地的：必须与 release 在同一线程获取。
        # auth.json 操作极小（<1ms），阻塞事件循环可接受（对齐 TS 同步实现）。
        lock.acquire()
        try:
            current = _read_text(self._path)
            result = await fn(current)
            next_value = result.get("next") if isinstance(result, dict) else None
            if next_value is not None:
                _write_text(self._path, next_value)
            return result.get("result") if isinstance(result, dict) else result
        finally:
            lock.release()


class InMemoryAuthStorageBackend:
    """内存后端（字符串内容保存在进程内）。"""

    def __init__(self, value: str | None = None) -> None:
        self._value = value

    def with_lock(self, fn: Any) -> Any:
        result = fn(self._value)
        next_value = result.get("next") if isinstance(result, dict) else None
        if next_value is not None:
            self._value = next_value
        return result.get("result") if isinstance(result, dict) else result

    async def with_lock_async(self, fn: Any) -> Any:
        result = await fn(self._value)
        next_value = result.get("next") if isinstance(result, dict) else None
        if next_value is not None:
            self._value = next_value
        return result.get("result") if isinstance(result, dict) else result


def _to_raw(credential: Credential) -> dict[str, Any]:
    if isinstance(credential, dict):
        return dict(credential)
    return {"type": credential.type, "key": credential.key}


def _from_raw(raw: dict[str, Any]) -> Credential:
    if raw.get("type") == "oauth":
        return raw
    return ApiKeyCredential(type="api_key", key=raw.get("key"))


def _parse_data(content: str | None) -> dict[str, Any]:
    if not content:
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AuthStorage:
    """文件/内存支持的 CredentialStore（含环境变量引用解析）。"""

    def __init__(self, storage: AuthStorageBackend) -> None:
        self._storage = storage
        self._data: dict[str, Any] = {}
        self.reload()

    @property
    def path(self) -> Path | None:
        """底层存储文件路径（内存后端为 None）。"""
        return getattr(self._storage, "_path", None)

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @staticmethod
    def create(auth_path: str | Path | None = None) -> "AuthStorage":
        return AuthStorage(FileAuthStorageBackend(auth_path))

    @staticmethod
    def from_storage(storage: AuthStorageBackend) -> "AuthStorage":
        return AuthStorage(storage)

    @staticmethod
    def in_memory(data: dict[str, Any] | None = None) -> "AuthStorage":
        storage = InMemoryAuthStorageBackend()
        storage.with_lock(
            lambda _current: {
                "result": None,
                "next": json.dumps(data or {}, ensure_ascii=False, indent=2),
            }
        )
        return AuthStorage.from_storage(storage)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """从存储重新加载内存快照（解析失败时保留最后有效快照）。"""
        try:
            self._data = self._storage.with_lock(lambda current: {"result": _parse_data(current)})
        except Exception:
            pass

    async def _read_credential(self, provider: str) -> Credential | None:
        credential = self._data.get(provider)
        if credential is None or not isinstance(credential, dict):
            return None
        if credential.get("type") != "api_key":
            return _from_raw(credential)
        key = credential.get("key")
        if not isinstance(key, str):
            return _from_raw(credential)
        env = credential.get("env") if isinstance(credential.get("env"), dict) else None
        resolved = resolve_config_value(key, env)
        if resolved is None:
            return None
        return ApiKeyCredential(type="api_key", key=resolved)

    # ------------------------------------------------------------------
    # CredentialStore 接口
    # ------------------------------------------------------------------

    async def read(self, provider: str) -> Credential | None:
        return await self._read_credential(provider)

    async def modify(
        self,
        provider: str,
        fn: Any,
    ) -> Credential | None:
        async def _locked(current: str | None) -> LockResult:
            current_data = _parse_data(current)
            current_credential = (
                _from_raw(current_data[provider])
                if isinstance(current_data.get(provider), dict)
                else None
            )
            next_value = await fn(current_credential)
            if next_value is None:
                self._data = current_data
                return {"result": current_data.get(provider), "next": None}
            merged = dict(current_data)
            merged[provider] = _to_raw(next_value)
            self._data = merged
            return {"result": next_value, "next": json.dumps(merged, ensure_ascii=False, indent=2)}

        return await self._storage.with_lock_async(_locked)

    async def delete(self, provider: str) -> None:
        async def _locked(current: str | None) -> LockResult:
            current_data = _parse_data(current)
            current_data.pop(provider, None)
            self._data = current_data
            return {"result": None, "next": json.dumps(current_data, ensure_ascii=False, indent=2)}

        await self._storage.with_lock_async(_locked)

    async def list(self) -> list[CredentialInfo]:
        return [
            {
                "provider_id": provider_id,
                "type": credential_type(credential) or "",
            }
            for provider_id, credential in self._data.items()
            if isinstance(credential, dict)
        ]

    # 兼容旧 API（Provider / 测试使用）。
    async def write(self, provider: str, credential: Credential) -> None:
        async def _set(_current: Credential | None) -> Credential:
            return credential

        await self.modify(provider, _set)

    def has_key(self, provider: str) -> bool:
        return provider in self._data


def read_stored_credential(
    provider_id: str,
    auth_path: str | Path | None = None,
) -> Credential | None:
    """一次性同步读取 auth.json 中的原始凭证（不解析配置值）。"""
    path = Path(auth_path) if auth_path else get_agent_dir() / "auth.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get(provider_id) if isinstance(data, dict) else None
    return _from_raw(raw) if isinstance(raw, dict) else None


def migrate_auth_to_auth_json() -> list[str]:
    """一次性迁移 oauth.json 与 settings.json apiKeys → auth.json（对齐 TS
    migrateAuthToAuthJson）。

    oauth.json 凭证包成 {"type": "oauth", ...}；settings.json 的 apiKeys
    包成 {"type": "api_key", "key": ...}；oauth.json 改名 .migrated；
    auth.json 以 0600 写入。返回迁移的 provider 列表。
    """
    agent_dir = get_agent_dir()
    auth_path = agent_dir / "auth.json"
    oauth_path = agent_dir / "oauth.json"
    settings_path = agent_dir / "settings.json"

    if auth_path.exists():
        return []

    migrated: dict[str, Any] = {}
    providers: list[str] = []

    if oauth_path.exists():
        try:
            oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
            if isinstance(oauth, dict):
                for provider, cred in oauth.items():
                    if isinstance(cred, dict):
                        migrated[provider] = {"type": "oauth", **cred}
                        providers.append(provider)
            oauth_path.rename(oauth_path.with_name("oauth.json.migrated"))
        except (OSError, ValueError):
            pass

    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(settings, dict):
                api_keys = settings.get("apiKeys")
                if isinstance(api_keys, dict):
                    for provider, key in api_keys.items():
                        if provider not in migrated and isinstance(key, str):
                            migrated[provider] = {"type": "api_key", "key": key}
                            providers.append(provider)
                    settings.pop("apiKeys", None)
                    settings_path.write_text(
                        json.dumps(settings, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
        except (OSError, ValueError):
            pass

    if migrated:
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = auth_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(auth_path)
        auth_path.chmod(0o600)

    return providers


__all__ = [
    "LockResult",
    "AuthStorageBackend",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "AuthStorage",
    "read_stored_credential",
    "migrate_auth_to_auth_json",
]
