"""telemetry 抽象测试。"""

from __future__ import annotations

import pytest

from pi_telemetry import (
    InMemoryTelemetryContext,
    NOOP_TELEMETRY_CONTEXT,
    SpanOptions,
    SpanStatus,
)


@pytest.mark.asyncio
async def test_noop_runs_callback() -> None:
    result = await NOOP_TELEMETRY_CONTEXT.start_span(
        SpanOptions(name="run"), lambda _span: _noop_result()
    )
    assert result == "ok"


async def _noop_result() -> str:
    return "ok"


@pytest.mark.asyncio
async def test_in_memory_records_span_events_and_status() -> None:
    context = InMemoryTelemetryContext()

    async def body(span) -> int:
        span.set_attributes({"model": "deepseek"})
        span.add_event("started")
        span.set_status(SpanStatus("ok"))
        return 42

    result = await context.start_span(SpanOptions(name="agent", attributes={"trace": "t1"}), body)
    assert result == 42
    assert len(context.spans) == 1
    record = context.spans[0]
    assert record.name == "agent"
    assert record.attributes == {"trace": "t1", "model": "deepseek"}
    assert record.events == [("started", {})]
    assert record.status is not None and record.status.status == "ok"
