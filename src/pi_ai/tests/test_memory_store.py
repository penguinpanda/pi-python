"""InMemoryMemoryStore 参考实现测试。"""

from __future__ import annotations

import pytest

from pi_ai.memory import InMemoryMemoryStore


@pytest.mark.asyncio
async def test_set_get_delete_and_copy_isolation():
    store = InMemoryMemoryStore()
    assert await store.get("k") is None

    await store.set("k", {"nested": [1, 2]})
    value = await store.get("k")
    value["nested"].append(3)
    assert (await store.get("k"))["nested"] == [1, 2]

    await store.delete("k")
    assert await store.get("k") is None


@pytest.mark.asyncio
async def test_search_matches_key_and_serialized_value():
    store = InMemoryMemoryStore()
    await store.set("alpha", {"text": "hello world"})
    await store.set("beta", {"text": "other"})

    assert await store.search("alpha") == [{"text": "hello world"}]
    assert await store.search("world") == [{"text": "hello world"}]
    assert await store.search("missing") == []
    assert await store.search("   ") == []


@pytest.mark.asyncio
async def test_search_respects_limit():
    store = InMemoryMemoryStore()
    await store.set("a1", {"text": "shared token"})
    await store.set("a2", {"text": "shared token"})
    await store.set("a3", {"text": "shared token"})

    assert len(await store.search("shared", limit=2)) == 2
