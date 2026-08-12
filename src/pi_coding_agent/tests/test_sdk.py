"""create_agent_session SDK 入口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent import AgentToolResult
from pi_coding_agent import (
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    create_agent_session,
)
from pi_coding_agent._session_manager_v4 import in_memory_session_manager
from pi_coding_agent.extensions import ToolDefinition
from pi_coding_agent.model_runtime import ModelRuntime
from pi_coding_agent.settings_manager import SettingsManager
from pi_ai.providers.faux import faux_provider


async def _faux_runtime() -> ModelRuntime:
    core = faux_provider()
    return await ModelRuntime.create(providers=[core.provider])


@pytest.mark.asyncio
async def test_create_agent_session_minimal() -> None:
    core = faux_provider()
    model = core.get_model()
    runtime = await ModelRuntime.create(providers=[core.provider])
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=".",
            model=model,
            model_runtime=runtime,
            settings_manager=SettingsManager.in_memory(),
            session_manager=await in_memory_session_manager("."),
        )
    )
    assert isinstance(result, CreateAgentSessionResult)
    assert result.session._cwd == str(Path(".").resolve())
    assert result.session._model is model
    assert result.session._agent.state.tools is not None


@pytest.mark.asyncio
async def test_create_agent_session_picks_available_model() -> None:
    runtime = await _faux_runtime()
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=".",
            model_runtime=runtime,
            settings_manager=SettingsManager.in_memory(),
            session_manager=await in_memory_session_manager("."),
        )
    )
    assert result.session._model.provider == "faux"


@pytest.mark.asyncio
async def test_create_agent_session_custom_tools_and_filters() -> None:
    core = faux_provider()
    model = core.get_model()
    runtime = await ModelRuntime.create(providers=[core.provider])

    async def _execute(tool_call_id, params, signal=None, on_update=None, context=None):
        return AgentToolResult(content=[{"type": "text", "text": "ok"}])

    custom = ToolDefinition(
        name="custom",
        description="custom tool",
        parameters={"type": "object", "properties": {}},
        execute=_execute,
    )
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=".",
            model=model,
            model_runtime=runtime,
            settings_manager=SettingsManager.in_memory(),
            session_manager=await in_memory_session_manager("."),
            no_tools="all",
            custom_tools=[custom],
            session_start_event={"source": "sdk"},
        )
    )
    tool_names = [tool.name for tool in result.session._agent.state.tools]
    assert tool_names == ["custom"]
