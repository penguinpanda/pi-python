"""Response stream golden 测试（faux provider 事件类型）。"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.types import Context


@pytest.mark.asyncio
async def test_text_stream_matches_golden_types() -> None:
    fixture = Path(__file__).parent / "fixtures" / "text-stream-types.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    core = faux_provider()
    core.set_responses([faux_assistant_message("ok")])
    model = core.get_model()
    stream = await core.stream(model, Context(messages=[{"role": "user", "content": "hi"}]))
    actual = [event["type"] for event in [event async for event in stream]]
    assert actual == expected
