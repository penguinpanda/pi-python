"""工具执行 parity golden 测试。"""

from __future__ import annotations

import pytest

from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)
from pi_ai.types import Model, TextContent, UserMessage
from pi_agent._agent_loop import run_agent_loop
from pi_agent._types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
)


def _tool(name: str, result_text: str) -> AgentTool:
    async def _execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(type="text", text=result_text)])

    return AgentTool(
        name=name,
        description=f"Tool: {name}",
        input_schema={"type": "object", "properties": {}},
        label=name,
        execute=_execute,
    )


@pytest.mark.asyncio
async def test_tool_execution_golden_order() -> None:
    model = Model(id="parity-model", provider="parity", api="openai-completions")
    core = faux_provider()
    core.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("search", {"q": "x"})],
                stop_reason="tool_call",
            ),
            faux_assistant_message("found"),
        ]
    )
    events: list[str] = []

    async def emit(event) -> None:  # type: ignore[no-untyped-def]
        events.append(event["type"])

    result = await run_agent_loop(
        [UserMessage(role="user", content="search x")],
        AgentContext(system_prompt="", messages=[], tools=[_tool("search", "result")]),
        AgentLoopConfig(model=model, convert_to_llm=lambda msgs: list(msgs)),
        emit=emit,
        signal=None,
        stream_fn=core.stream,
    )

    start = events.index("tool_execution_start")
    end = events.index("tool_execution_end")
    assert start < end
    assert any(message.get("role") == "toolResult" for message in result)
