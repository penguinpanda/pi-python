"""Agent loop parity golden 测试。"""

from __future__ import annotations

import pytest

from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.types import Model, UserMessage
from pi_agent._agent_loop import run_agent_loop
from pi_agent._types import AgentContext, AgentLoopConfig


@pytest.mark.asyncio
async def test_single_turn_event_golden_sequence() -> None:
    model = Model(id="parity-model", provider="parity", api="openai-completions")
    core = faux_provider()
    core.set_responses([faux_assistant_message("ok")])
    events: list[str] = []

    async def emit(event) -> None:  # type: ignore[no-untyped-def]
        events.append(event["type"])

    await run_agent_loop(
        [UserMessage(role="user", content="hi")],
        AgentContext(system_prompt="", messages=[]),
        AgentLoopConfig(model=model, convert_to_llm=lambda msgs: list(msgs)),
        emit=emit,
        signal=None,
        stream_fn=core.stream,
    )

    assert events == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "auto_retry_end",
        "message_end",
        "turn_end",
        "agent_end",
    ]
