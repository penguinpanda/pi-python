"""AuthStorage 单元测试（文件 + 内存后端、env 引用解析、并发锁）。"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from pi_ai.auth import ApiKeyCredential

from pi_coding_agent.auth_storage import (
    AuthStorage,
    FileAuthStorageBackend,
    migrate_auth_to_auth_json,
    read_stored_credential,
)


def test_migrate_auth_to_auth_json(tmp_path, monkeypatch):
    """oauth.json + settings.json apiKeys → auth.json（0600）；oauth.json 改名 .migrated。"""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    (tmp_path / "oauth.json").write_text(
        json.dumps({"github": {"token": "gho_x", "account": "me"}}), encoding="utf-8"
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"apiKeys": {"openai": "sk-abc", "deepseek": "sk-def"}, "other": 1}),
        encoding="utf-8",
    )

    providers = migrate_auth_to_auth_json()

    assert set(providers) == {"github", "openai", "deepseek"}
    auth = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    assert auth["github"] == {"type": "oauth", "token": "gho_x", "account": "me"}
    assert auth["openai"] == {"type": "api_key", "key": "sk-abc"}
    assert auth["deepseek"] == {"type": "api_key", "key": "sk-def"}
    assert (tmp_path / "auth.json").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "oauth.json").exists()
    assert (tmp_path / "oauth.json.migrated").exists()
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert "apiKeys" not in settings
    assert settings["other"] == 1

    # 已存在 auth.json 时跳过
    assert migrate_auth_to_auth_json() == []


class TestInMemoryAuthStorage:
    async def test_modify_read_delete(self):
        store = AuthStorage.in_memory()

        async def _set(_current):
            return ApiKeyCredential(key="sk-test")

        await store.modify("deepseek", _set)

        credential = await store.read("deepseek")
        assert credential is not None
        assert credential.key == "sk-test"

        infos = await store.list()
        assert infos == [{"provider_id": "deepseek", "type": "api_key"}]

        await store.delete("deepseek")
        assert await store.read("deepseek") is None
        assert await store.list() == []

    async def test_modify_with_none_result_keeps_current(self):
        store = AuthStorage.in_memory()

        async def _set(_current):
            return ApiKeyCredential(key="sk-a")

        await store.modify("p", _set)

        async def _unset(_current):
            return None

        result = await store.modify("p", _unset)
        assert result is not None
        credential = await store.read("p")
        assert credential.key == "sk-a"

    async def test_read_resolves_env_reference(self, monkeypatch):
        monkeypatch.setenv("TEST_PI_STORED_KEY", "sk-env-resolved")
        store = AuthStorage.in_memory(
            {"deepseek": {"type": "api_key", "key": "$TEST_PI_STORED_KEY"}}
        )
        credential = await store.read("deepseek")
        assert credential.key == "sk-env-resolved"


class TestFileAuthStorage:
    def _store(self, tmp_path):
        return AuthStorage(FileAuthStorageBackend(tmp_path / "auth.json"))

    async def test_persists_across_instances(self, tmp_path):
        store1 = self._store(tmp_path)

        async def _set(_current):
            return ApiKeyCredential(key="sk-file")

        await store1.modify("openai", _set)

        store2 = self._store(tmp_path)
        credential = await store2.read("openai")
        assert credential.key == "sk-file"
        assert store2.path is not None
        assert store2.path.exists()

    async def test_file_created_on_write(self, tmp_path):
        store = self._store(tmp_path)

        async def _set(_current):
            return ApiKeyCredential(key="sk-lock")

        await store.modify("openai", _set)
        # filelock 在 Windows 释放后会自动清理 .lock 文件；
        # 有意义的断言是 auth.json 本身已创建。
        assert (tmp_path / "auth.json").exists()

    async def test_delete_removes_entry(self, tmp_path):
        store = self._store(tmp_path)

        async def _set(_current):
            return ApiKeyCredential(key="sk-del")

        await store.modify("openai", _set)
        await store.delete("openai")
        assert await store.read("openai") is None

    async def test_concurrent_modify_serialized(self, tmp_path):
        store = self._store(tmp_path)

        async def _make(key: str):
            return ApiKeyCredential(key=key)

        async def _write(key: str):
            await store.modify("p", lambda _current: _make(key))

        await asyncio.gather(*(_write(f"sk-{i}") for i in range(5)))
        data = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        assert data["p"]["key"] in {f"sk-{i}" for i in range(5)}


class TestReadStoredCredential:
    def test_reads_raw_credential(self, tmp_path):
        path = tmp_path / "auth.json"
        path.write_text(
            json.dumps({"deepseek": {"type": "api_key", "key": "sk-raw"}}),
            encoding="utf-8",
        )
        credential = read_stored_credential("deepseek", path)
        assert credential is not None
        assert credential.key == "sk-raw"

    def test_missing_file_returns_none(self, tmp_path):
        assert read_stored_credential("deepseek", tmp_path / "nope.json") is None


@pytest.mark.skipif(sys.platform == "win32", reason="posix file modes only")
class TestAuthFilePermissions:
    @staticmethod
    def _store(tmp_path) -> AuthStorage:
        return AuthStorage(FileAuthStorageBackend(tmp_path / "auth.json"))

    async def test_write_sets_0600_on_new_file(self, tmp_path):
        """新写入的 auth.json 权限必须为 0600（对齐 TS mode: 0o600）。"""
        store = self._store(tmp_path)

        async def _set(_current):
            return ApiKeyCredential(key="sk-file")

        await store.modify("openai", _set)
        path = tmp_path / "auth.json"
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600

    async def test_write_converges_preexisting_loose_permissions(self, tmp_path):
        """已存在的宽松权限文件（0644）写入后收敛为 0600（对齐 TS chmodSync）。"""
        path = tmp_path / "auth.json"
        path.write_text("{}", encoding="utf-8")
        path.chmod(0o644)
        store = self._store(tmp_path)

        async def _set(_current):
            return ApiKeyCredential(key="sk-file")

        await store.modify("openai", _set)
        assert (path.stat().st_mode & 0o777) == 0o600
