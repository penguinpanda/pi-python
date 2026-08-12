"""Agent 内联 Telemetry Schema 测试。"""

from __future__ import annotations

import pytest

from pi_agent.telemetry_schema import (
    AGENT_TELEMETRY_SCHEMAS,
    AI_TELEMETRY_SCHEMA,
    HARNESS_TELEMETRY_SCHEMA,
    start_ai_span,
    start_harness_span,
)
from pi_telemetry import InMemoryTelemetryContext


def test_ai_telemetry_schema_import_and_shape() -> None:
    assert AI_TELEMETRY_SCHEMA["version"] == 1
    spans = AI_TELEMETRY_SCHEMA["spans"]
    assert set(spans) == {"pi.ai.request"}
    request = spans["pi.ai.request"]
    assert "startAttributes" in request
    assert "endAttributes" in request
    assert request["status"]["default"] == "ok"
    for required in (
        "pi.ai.operation",
        "pi.ai.provider",
        "pi.ai.model",
        "pi.ai.api",
        "pi.ai.streaming",
    ):
        assert request["startAttributes"][required]["required"] is True
    assert request["startAttributes"]["pi.ai.operation"]["values"] == [
        "stream",
        "fetch_deferred",
        "cancel_deferred",
        "generate_images",
    ]
    assert request["endAttributes"]["pi.ai.response.stop_reason"]["values"] == [
        "stop",
        "length",
        "tool_use",
        "error",
        "aborted",
        "deferred",
    ]


def test_harness_telemetry_schema_spans() -> None:
    spans = HARNESS_TELEMETRY_SCHEMA["spans"]
    expected = {
        "pi.harness.run",
        "pi.harness.compaction",
        "pi.harness.navigation",
        "pi.harness.checkpoint",
        "pi.harness.turn",
        "pi.harness.step",
        "pi.harness.tool",
        "pi.harness.hook",
        "pi.harness.sleep",
        "pi.harness.event_handler",
        "pi.session.write",
    }
    assert set(spans) == expected
    for span in spans.values():
        assert "startAttributes" in span
        assert "endAttributes" in span
        assert "status" in span


def test_agent_schemas_combined() -> None:
    assert len(AGENT_TELEMETRY_SCHEMAS) == 2
    assert AGENT_TELEMETRY_SCHEMAS[0] is AI_TELEMETRY_SCHEMA
    assert AGENT_TELEMETRY_SCHEMAS[1] is HARNESS_TELEMETRY_SCHEMA


@pytest.mark.asyncio
async def test_start_ai_span_runs_callback() -> None:
    context = InMemoryTelemetryContext()

    async def _body(span) -> str:
        span.set_attributes({"pi.ai.streaming": True})
        return "ok"

    result = await start_ai_span(
        context,
        {
            "pi.ai.operation": "stream",
            "pi.ai.provider": "faux",
            "pi.ai.model": "faux-1",
            "pi.ai.api": "openai-completions",
            "pi.ai.streaming": True,
        },
        _body,
    )
    assert result == "ok"
    assert context.spans[0].name == "pi.ai.request"


@pytest.mark.asyncio
async def test_start_harness_span_runs_callback() -> None:
    context = InMemoryTelemetryContext()

    async def _body(span) -> int:
        return 7

    result = await start_harness_span(context, "pi.harness.compaction", {}, _body)
    assert result == 7
    assert context.spans[0].name == "pi.harness.compaction"
