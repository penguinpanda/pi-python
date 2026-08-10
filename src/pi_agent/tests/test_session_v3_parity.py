"""v3 Session 与 TS v0.84 的小缺口对齐测试。"""

from __future__ import annotations

import pytest

from pi_agent.session import InMemorySessionStorage, Session

from test_session_v4_conformance import (
    create_assistant_message,
    create_user_message,
)


@pytest.mark.asyncio
async def test_deferred_assistant_message_skipped_in_context():
    """对齐 TS v0.84：deferred assistant 消息不进 LLM 上下文。"""
    session = Session(InMemorySessionStorage())
    deferred = create_assistant_message("deferred")
    deferred["stopReason"] = "deferred"
    await session.append_message(deferred)
    await session.append_message(create_user_message("root"))

    context = await session.build_context()
    assert [message["role"] for message in context["messages"]] == ["user"]
