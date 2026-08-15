"""凭证存储（InMemory / File）测试。"""

import asyncio
import json
import sys

import pytest

from pi_ai.auth import ApiKeyCredential
from pi_ai.auth.credential_store import (
    CredentialStoreCorruptError,
    FileCredentialStore,
    InMemoryCredentialStore,
)


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
    # 损坏文件被备份而非被覆盖。
    assert (tmp_path / "auth.json.corrupt").exists()


@pytest.mark.asyncio
async def test_file_store_corrupt_write_raises(tmp_path):
    """损坏文件上执行写入必须报错，不得静默清空全部凭证。"""
    path = tmp_path / "auth.json"
    path.write_text("{not json")
    store = FileCredentialStore(path)

    async def _set(_current):
        return {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}

    with pytest.raises(CredentialStoreCorruptError):
        await store.write("p", _set)
    assert (tmp_path / "auth.json.corrupt").exists()


@pytest.mark.asyncio
async def test_file_store_concurrent_cross_provider_no_lost_update(tmp_path):
    """不同 provider 的并发 modify 不得互相覆盖（读-改-写全程串行化）。"""
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)
    await store.write("a", ApiKeyCredential(key="key-a"))
    await store.write("b", ApiKeyCredential(key="key-b"))

    async def fn_a(_current):
        await asyncio.sleep(0.05)
        return ApiKeyCredential(key="key-a-updated")

    async def fn_b(_current):
        await asyncio.sleep(0.01)
        return ApiKeyCredential(key="key-b-updated")

    await asyncio.gather(store.modify("a", fn_a), store.modify("b", fn_b))

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["a"]["key"] == "key-a-updated"
    assert raw["b"]["key"] == "key-b-updated"


@pytest.mark.asyncio
async def test_file_store_atomic_write_no_tmp_leftovers(tmp_path):
    """原子写不得在目录中残留临时文件。"""
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)
    for i in range(5):
        await store.write(f"p{i}", ApiKeyCredential(key=f"k{i}"))
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name not in ("auth.json", "auth.json.lock")
    ]
    assert leftovers == []


@pytest.mark.skipif(sys.platform == "win32", reason="posix file modes only")
@pytest.mark.asyncio
async def test_file_store_writes_0600(tmp_path):
    """凭证文件写入后权限必须为 0600（对齐 TS chmodSync）。"""
    path = tmp_path / "auth.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)  # 预置宽松权限，验证写入后收敛
    store = FileCredentialStore(path)
    await store.write("openai", ApiKeyCredential(key="sk-file"))
    assert (path.stat().st_mode & 0o777) == 0o600
