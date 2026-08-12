"""Session v4 parity golden 测试。"""

from __future__ import annotations

import pytest

from pi_agent.session.v4.memory import InMemorySessionRepo


def _zero_usage() -> dict:
    return {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total": 0,
        },
    }


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _assistant(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "openai-completions",
        "provider": "parity",
        "model": "parity-model",
        "usage": _zero_usage(),
        "stopReason": "stop",
        "timestamp": 1,
    }


@pytest.mark.asyncio
async def test_session_v4_context_golden() -> None:
    repo = InMemorySessionRepo()
    session = await repo.create({"id": "s"})
    await session.append_message(_user("root"))
    await session.append_message(_assistant("shared"))
    await session.append_message(_user("tail"))

    context = await session.build_context()
    stats = await session.get_stats()

    assert [message["role"] for message in context["messages"]] == ["user", "assistant", "user"]
    assert context["messages"][0]["content"] == [{"type": "text", "text": "root"}]
    assert context["messages"][1]["content"] == [{"type": "text", "text": "shared"}]
    assert context["messages"][2]["content"] == [{"type": "text", "text": "tail"}]
    assert stats["messageCount"] == 3
