"""压缩 / 分支摘要使用主模型（对齐 TS：无独立摘要模型）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai._types import UserMessage
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage


def _make_runtime(record: dict):
    models = Models(credentials=AuthStorage.in_memory())
    models_list = [
        Model(id="faux-1", provider="faux", api="openai-completions"),
        Model(id="faux-2", provider="faux", api="openai-completions"),
    ]
    core = faux_provider(models=models_list)

    async def factory(context, _options, state, model):
        record["model_id"] = model.id
        return faux_assistant_message("## Goal\nsummary text")

    core.set_responses([factory])
    models.add_provider(core.provider)
    return models


def _make_session(models: Models, tmp_path: Path, manager: SessionManager) -> AgentSession:
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=models.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
    )


@pytest.mark.asyncio
async def test_branch_summary_uses_main_model(tmp_path):
    """回归：分支摘要使用主模型（对齐 TS，无独立摘要模型）。"""
    record: dict = {}
    models = _make_runtime(record)
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="branch"))
    await manager.append_message(UserMessage(role="user", content="more"))

    session = _make_session(models, tmp_path, manager)
    try:
        await session.navigate_to(e1)
        assert record.get("model_id") == "faux-1"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_compaction_uses_main_model(tmp_path):
    """回归：压缩摘要使用主模型（对齐 TS，无独立摘要模型）。"""
    record: dict = {}
    models = _make_runtime(record)
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    for _ in range(40):
        await manager.append_message(UserMessage(role="user", content="x" * 3000))

    session = _make_session(models, tmp_path, manager)
    try:
        result = await session.compact()
        assert result is not None
        assert record.get("model_id") == "faux-1"
    finally:
        await session.dispose()
