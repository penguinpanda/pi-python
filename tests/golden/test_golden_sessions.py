"""Session v4 JSONL golden 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent.session.v4.repo import JsonlSessionRepo


@pytest.mark.asyncio
async def test_golden_session_loads_and_builds_context(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "session-v4.jsonl"
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    session = await repo.open_path(fixture)

    context = await session.build_context()
    stats = await session.get_stats()

    assert [message["role"] for message in context["messages"]] == ["user", "assistant"]
    assert context["messages"][0]["content"] == [{"type": "text", "text": "root"}]
    assert context["messages"][1]["content"] == [{"type": "text", "text": "shared"}]
    assert stats["messageCount"] == 2
