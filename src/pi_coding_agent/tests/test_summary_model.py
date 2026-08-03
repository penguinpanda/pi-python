"""独立摘要模型配置测试。"""

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


def _make_session(
    models: Models, tmp_path: Path, manager: SessionManager, summary_model
) -> AgentSession:
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
        summary_model=summary_model,
    )


@pytest.mark.asyncio
async def test_branch_summary_uses_summary_model(tmp_path):
    record: dict = {}
    models = _make_runtime(record)
    summary_model = models.get_model("faux", "faux-2")
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="branch"))
    await manager.append_message(UserMessage(role="user", content="more"))

    session = _make_session(models, tmp_path, manager, summary_model)
    try:
        await session.navigate_to(e1)
        assert record.get("model_id") == "faux-2"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_branch_summary_uses_main_model_by_default(tmp_path):
    record: dict = {}
    models = _make_runtime(record)
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    e1 = await manager.append_message(UserMessage(role="user", content="branch"))
    await manager.append_message(UserMessage(role="user", content="more"))

    session = _make_session(models, tmp_path, manager, summary_model=None)
    try:
        await session.navigate_to(e1)
        assert record.get("model_id") == "faux-1"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_compaction_uses_summary_model(tmp_path):
    record: dict = {}
    models = _make_runtime(record)
    summary_model = models.get_model("faux", "faux-2")
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    # 足够大的上下文，让 keep_recent_tokens 预算被超过、产生待摘要消息。
    for _ in range(40):
        await manager.append_message(UserMessage(role="user", content="x" * 3000))

    session = _make_session(models, tmp_path, manager, summary_model)
    try:
        result = await session.compact()
        assert result is not None
        assert record.get("model_id") == "faux-2"
    finally:
        await session.dispose()
