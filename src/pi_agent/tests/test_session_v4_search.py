"""v4 会话搜索测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.search import ScanningSessionSearch

from test_session_v4_conformance import create_user_message


@pytest.fixture
def repo(tmp_path: Path) -> JsonlSessionRepo:
    return JsonlSessionRepo(str(tmp_path / "sessions"))


@pytest.mark.asyncio
async def test_search_finds_text_in_entries(repo, tmp_path):
    session = await repo.create({"id": "one", "cwd": str(tmp_path)})
    await session.append_message(create_user_message("hello world"))
    await session.set_name("My Session")

    search = ScanningSessionSearch(repo)
    hits = await search.search({"text": "hello"})
    assert len(hits) == 1
    assert hits[0]["entryId"] == (await session.find_entries())[0]["id"]
    assert hits[0]["metadata"]["id"] == "one"
    assert "hello world" in hits[0]["snippet"]


@pytest.mark.asyncio
async def test_search_filters_by_cwd(repo, tmp_path):
    session_a = await repo.create({"id": "a", "cwd": str(tmp_path / "p1")})
    await session_a.append_message(create_user_message("needle"))
    session_b = await repo.create({"id": "b", "cwd": str(tmp_path / "p2")})
    await session_b.append_message(create_user_message("needle"))

    search = ScanningSessionSearch(repo)
    hits = await search.search({"text": "needle", "cwd": str(tmp_path / "p1")})
    assert [hit["metadata"]["id"] for hit in hits] == ["a"]


@pytest.mark.asyncio
async def test_search_empty_query_returns_nothing(repo, tmp_path):
    await repo.create({"id": "a", "cwd": str(tmp_path)})
    search = ScanningSessionSearch(repo)
    assert await search.search({"text": "   "}) == []
    assert await search.search() == []
