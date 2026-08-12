"""Models 请求级 telemetry span 测试。"""

from __future__ import annotations

import pytest

from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.types import Context
from pi_telemetry import InMemoryTelemetryContext


@pytest.mark.asyncio
async def test_models_stream_records_request_span() -> None:
    core = faux_provider()
    core.set_responses([faux_assistant_message("ok")])
    models = Models()
    models.add_provider(core.provider)
    telemetry = InMemoryTelemetryContext()
    stream = await models.stream(
        core.get_model(),
        Context(messages=[{"role": "user", "content": "hi"}]),
        {"telemetry_context": telemetry},
    )
    await stream.result()
    assert any(span.name == "pi.ai.request" for span in telemetry.spans)
