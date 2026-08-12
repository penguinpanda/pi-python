"""EventStream parity golden 测试。"""

from __future__ import annotations

import pytest

from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)
from pi_ai.types import Context


async def _event_types(message) -> list[str]:  # type: ignore[no-untyped-def]
    core = faux_provider()
    core.set_responses([message])
    model = core.get_model()
    stream = await core.stream(
        model,
        Context(messages=[{"role": "user", "content": "hi"}]),
    )
    return [event["type"] for event in [event async for event in stream]]


@pytest.mark.asyncio
async def test_text_stream_golden_sequence() -> None:
    assert await _event_types(faux_assistant_message("ok")) == [
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.asyncio
async def test_tool_stream_golden_sequence() -> None:
    message = faux_assistant_message(
        [faux_tool_call("read", {"path": "a.txt"})],
        stop_reason="tool_call",
    )
    assert await _event_types(message) == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
