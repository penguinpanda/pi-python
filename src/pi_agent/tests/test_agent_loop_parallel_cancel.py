"""并行工具批次 prepare 阶段异常的任务取消回归测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import Model, TextContent

from pi_agent._agent_loop import _execute_tool_calls_parallel
from pi_agent._types import AgentContext, AgentEvent, AgentLoopConfig, AgentTool, AgentToolResult


def _make_model() -> Model:
    return Model(id="test-model", provider="test", api="openai-completions", name="Test")


def _make_tool(name: str, execute) -> AgentTool:
    return AgentTool(
        name=name,
        description=f"Tool: {name}",
        input_schema={"type": "object", "properties": {}},
        label=name,
        execute=execute,
    )


@pytest.mark.asyncio
async def test_parallel_prepare_failure_cancels_started_tasks() -> None:
    """第二批 prepare 抛异常时,第一批已启动的工具任务必须被取消。"""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocker_execute(tool_call_id, params, signal=None, on_update=None):
        started.set()
        try:
            await asyncio.Event().wait()  # 永久阻塞,仅取消可结束
        finally:
            cancelled.set()
        return AgentToolResult(content=[TextContent(type="text", text="never")])

    async def raiser_execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(type="text", text="never")])

    async def before(ctx):
        if ctx.tool_call["name"] == "raiser":
            # 保证 blocker 已进入执行后再抛,验证取消路径。
            await asyncio.wait_for(started.wait(), timeout=5)
            raise ValueError("hook boom")
        return None

    blocker = _make_tool("blocker", blocker_execute)
    raiser = _make_tool("raiser", raiser_execute)
    config = AgentLoopConfig(
        model=_make_model(),
        convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        before_tool_call=before,
    )
    context = AgentContext(system_prompt="", messages=[], tools=[blocker, raiser])
    assistant = faux_assistant_message(
        [
            faux_tool_call("blocker", {}, tool_call_id="tc-1"),
            faux_tool_call("raiser", {}, tool_call_id="tc-2"),
        ],
        stop_reason="tool_call",
    )
    tool_calls = [block for block in assistant["content"] if block["type"] == "toolCall"]

    async def emit(evt: AgentEvent) -> None:
        pass

    with pytest.raises(ValueError, match="hook boom"):
        await _execute_tool_calls_parallel(tool_calls, assistant, context, config, emit, None)

    assert cancelled.is_set()
