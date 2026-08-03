"""AgentSession.navigate_to 分支导航测试（含分支摘要）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai._types import UserMessage
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage


def _make_session(tmp_path: Path, manager: SessionManager) -> AgentSession:
    models = Models(credentials=AuthStorage.in_memory())
    core = faux_provider()
    core.set_responses([
        faux_assistant_message("## Goal\nSummary of the other branch"),
        faux_assistant_message("ok"),
    ])
    models.add_provider(core.provider)
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=models.stream,
    ))
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
    )


@pytest.mark.asyncio
async def test_navigate_to_generates_branch_summary(tmp_path):
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="branch A"))
    await manager.append_message(UserMessage(role="user", content="branch A cont"))

    session = _make_session(tmp_path, manager)
    try:
        moved = await session.navigate_to(e1, summarize=True)
        assert moved is True

        entries = manager.get_entries()
        assert entries[-1]["type"] == "branch_summary"
        assert "Summary of the other branch" in entries[-1]["summary"]
        assert manager.get_leaf_id() == entries[-1]["id"]

        # agent 上下文已重建为 e1 分支。
        assert [m.get("content") for m in session.get_messages()] == ["branch A"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_navigate_to_without_summary(tmp_path):
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="base"))
    e2 = await manager.append_message(UserMessage(role="user", content="tail"))

    session = _make_session(tmp_path, manager)
    try:
        moved = await session.navigate_to(e1, summarize=False)
        assert moved is True
        assert manager.get_leaf_id() == e1
        assert [m.get("content") for m in session.get_messages()] == ["base"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_navigate_to_same_entry_noop(tmp_path):
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="only"))
    session = _make_session(tmp_path, manager)
    try:
        assert await session.navigate_to(e1) is False
    finally:
        await session.dispose()
