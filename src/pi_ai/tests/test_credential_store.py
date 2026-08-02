"""凭证存储（InMemory / File）测试。"""

import asyncio

import pytest

from pi_ai.auth import ApiKeyCredential
from pi_ai.auth.credential_store import FileCredentialStore, InMemoryCredentialStore


@pytest.mark.asyncio
async def test_in_memory_write_read():
    store = InMemoryCredentialStore()
    await store.write("openai", ApiKeyCredential(key="sk-1"))
    cred = await store.read("openai")
    assert cred is not None
    assert cred.key == "sk-1"


@pytest.mark.asyncio
async def test_in_memory_modify_serialized():
    store = InMemoryCredentialStore()
    seen = []

    async def fn1(_current):
        await asyncio.sleep(0.01)
        seen.append("fn1")
        return {"type": "oauth", "access": "a1", "refresh": "r1", "expires": 1}

    async def fn2(_current):
        seen.append("fn2")
        return None  # 不修改

    await asyncio.gather(store.modify("p", fn1), store.modify("p", fn2))
    assert seen == ["fn1", "fn2"]
    cred = await store.read("p")
    assert cred["access"] == "a1"


@pytest.mark.asyncio
async def test_in_memory_list():
    store = InMemoryCredentialStore()
    await store.write("a", ApiKeyCredential(key="k"))
    await store.write("b", {"type": "oauth", "access": "x", "refresh": "y", "expires": 0})
    infos = await store.list()
    by_id = {info["provider_id"]: info["type"] for info in infos}
    assert by_id == {"a": "api_key", "b": "oauth"}


@pytest.mark.asyncio
async def test_file_store_roundtrip(tmp_path):
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)
    await store.write("openai", ApiKeyCredential(key="sk-file"))
    oauth = {"type": "oauth", "access": "a", "refresh": "r", "expires": 123}

    async def _set(_current):
        return oauth

    await store.modify("openrouter", _set)
    assert await store.read("openai") is not None
    assert (await store.read("openai")).key == "sk-file"
    assert (await store.read("openrouter"))["access"] == "a"
    assert path.exists()
    # 重新打开（模拟跨进程持久化）
    store2 = FileCredentialStore(path)
    assert (await store2.read("openrouter"))["refresh"] == "r"
    await store2.delete("openai")
    assert await store2.read("openai") is None


@pytest.mark.asyncio
async def test_file_store_list(tmp_path):
    store = FileCredentialStore(tmp_path / "auth.json")
    await store.write("p", {"type": "oauth", "access": "a", "refresh": "r", "expires": 1})
    infos = await store.list()
    assert infos == [{"provider_id": "p", "type": "oauth"}]


@pytest.mark.asyncio
async def test_file_store_corrupt(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json")
    store = FileCredentialStore(path)
    assert await store.read("p") is None
    assert await store.list() == []
