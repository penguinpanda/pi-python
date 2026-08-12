"""v4 持久化搜索索引测试（P3）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.search_index import (
    PersistentSessionSearchIndex,
    rebuild_v4_search_index,
)

from pi_agent.session.v4.testing.conformance import create_user_message


@pytest.fixture
def repo(tmp_path: Path) -> JsonlSessionRepo:
    return JsonlSessionRepo(str(tmp_path / "sessions"))


@pytest.mark.asyncio
async def test_rebuild_and_search(repo, tmp_path):
    session_a = await repo.create({"id": "a", "cwd": str(tmp_path)})
    await session_a.append_message(create_user_message("hello world"))
    await session_a.set_name("Alpha")
    session_b = await repo.create({"id": "b", "cwd": str(tmp_path / "other")})
    await session_b.append_message(create_user_message("hello there"))

    index = PersistentSessionSearchIndex(str(tmp_path / "index.json"))
    await rebuild_v4_search_index(repo, index)

    hits = await index.search({"text": "hello"})
    assert {hit["metadata"]["id"] for hit in hits} == {"a", "b"}
    hits_cwd = await index.search({"text": "hello", "cwd": str(tmp_path)})
    assert [hit["metadata"]["id"] for hit in hits_cwd] == ["a"]
    assert await index.search({"text": "   "}) == []


@pytest.mark.asyncio
async def test_upsert_delete_and_persistence(repo, tmp_path):
    session = await repo.create({"id": "a", "cwd": str(tmp_path)})
    index = PersistentSessionSearchIndex(str(tmp_path / "index.json"))

    entry_id = await session.append_message(create_user_message("needle"))
    metadata = await session.get_metadata()
    entry = await session.get_entry(entry_id)
    assert entry is not None
    await index.upsert_entry(metadata, entry)

    hits = await index.search({"text": "needle"})
    assert len(hits) == 1
    assert hits[0]["entryId"] == entry_id

    reloaded = PersistentSessionSearchIndex(str(tmp_path / "index.json"))
    assert len(await reloaded.search({"text": "needle"})) == 1

    await index.delete_session(metadata)
    assert await index.search({"text": "needle"}) == []


@pytest.mark.asyncio
async def test_replace_session(repo, tmp_path):
    session = await repo.create({"id": "a", "cwd": str(tmp_path)})
    await session.append_message(create_user_message("old"))
    index = PersistentSessionSearchIndex(str(tmp_path / "index.json"))
    metadata = await session.get_metadata()
    entries = await session.find_entries({"order": "oldestFirst"})
    await index.replace_session(metadata, entries)
    assert len(await index.search({"text": "old"})) == 1

    await session.append_message(create_user_message("new"))
    await index.replace_session(metadata, await session.find_entries({"order": "oldestFirst"}))
    assert len(await index.search({"text": "old"})) == 1
    assert len(await index.search({"text": "new"})) == 1
