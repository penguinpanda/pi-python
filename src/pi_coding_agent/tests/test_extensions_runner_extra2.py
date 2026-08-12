"""ExtensionRunner 动作与工具同步补充测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pi_coding_agent.extensions.runner import ExtensionRunner

from pi_coding_agent.tests.test_extensions_runner_extra import _FakeSession


def test_set_active_tools_updates_session() -> None:
    session = _FakeSession()
    tool = SimpleNamespace(
        name="read",
        description="d",
        input_schema={},
        prompt_guidelines="g",
    )
    session._agent.state.tools = [tool]

    runner = ExtensionRunner()
    runner._set_active_tools(session, ["read"])
    assert session._agent.state.tools == [tool]
    assert session.extension_state["active_tools"] == [tool]


@pytest.mark.asyncio
async def test_send_user_message_delivery_modes() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()

    runner._action_send_user_message(session, "steer text", {"deliverAs": "steer"})
    await asyncio.sleep(0.05)
    assert session.steer_calls == ["steer text"]

    runner._action_send_user_message(session, "follow text", {"deliverAs": "followUp"})
    await asyncio.sleep(0.05)
    assert session.follow_up_calls == ["follow text"]

    runner._action_send_user_message(session, "prompt text", {})
    await asyncio.sleep(0.05)
    assert session.prompt_calls == ["prompt text"]


@pytest.mark.asyncio
async def test_action_set_model_with_bound_session() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()
    runner.bind_session(session)

    model = object()
    assert await runner._action_set_model(model) is True
    assert session.model is model


@pytest.mark.asyncio
async def test_command_action_success() -> None:
    runner = ExtensionRunner()
    runner.bind(command_handlers={"do": lambda: "ok"})
    assert await runner._command_action("do") == "ok"
