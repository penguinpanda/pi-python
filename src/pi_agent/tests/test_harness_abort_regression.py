"""AgentHarness abort 路径回归测试（不重复事件、不合成空消息）。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from test_harness import _make_harness


@pytest.mark.asyncio
async def test_abort_single_agent_end_and_no_synthetic_message() -> None:
    """abort 后 prompt 返回真实 aborted 消息；agent_end/settled 各一次；
    会话不落空失败消息。"""
    core = faux_provider(tokens_per_second=100)
    core.set_responses([faux_assistant_message("A" * 400)])
    harness = _make_harness(stream_fn=core.stream)

    events: list[str] = []
    harness.subscribe(lambda e, signal: events.append(e["type"]))

    run_task = asyncio.create_task(harness.prompt("hi"))
    await asyncio.sleep(0.05)
    await harness.abort()
    result = await asyncio.wait_for(run_task, timeout=5.0)

    assert result.get("stop_reason") == "aborted"
    assert events.count("agent_end") == 1
    assert events.count("settled") == 1

    context = await harness._session.build_context()
    assistants = [m for m in context["messages"] if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].get("stop_reason") == "aborted"
