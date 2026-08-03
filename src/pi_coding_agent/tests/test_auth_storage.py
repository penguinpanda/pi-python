"""AuthStorage 单元测试（文件 + 内存后端、env 引用解析、并发锁）。"""

from __future__ import annotations

import asyncio
import json

from pi_ai.auth import ApiKeyCredential

from pi_coding_agent.auth_storage import (
    AuthStorage,
    FileAuthStorageBackend,
    InMemoryAuthStorageBackend,
    read_stored_credential,
)


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
