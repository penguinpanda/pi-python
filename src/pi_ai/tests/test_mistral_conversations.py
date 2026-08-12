"""Mistral Conversations API 测试。"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest

from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.mistral_conversations import (
    create_mistral_tool_call_id_normalizer,
    mistral_stream,
)
from pi_ai.types import AssistantMessage, Context, Model
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model() -> Model:
    return Model(id="mistral-medium-3.5", provider="mistral", api="mistral-conversations")


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi", "timestamp": 0}])


def test_mistral_conversations_api_registered() -> None:
    assert get_api_provider("mistral-conversations") is not None


def test_mistral_tool_call_id_normalizer() -> None:
    normalize = create_mistral_tool_call_id_normalizer()
    source: AssistantMessage = {
        "role": "assistant",
        "api": "mistral-conversations",
        "provider": "mistral",
        "model": "mistral-medium-3.5",
        "timestamp": 0,
        "content": [],
    }
    first = normalize("call_1|fc_item_very_long_that_mistral_rejects", _model(), source)
    second = normalize("call_1|fc_item_very_long_that_mistral_rejects", _model(), source)
    other = normalize("another-tool-call-id", _model(), source)
    assert re.fullmatch(r"[a-zA-Z0-9]{9}", first)
    assert first == second
    assert first != other


@pytest.mark.asyncio
async def test_mistral_stream_delegates_to_completions() -> None:
    captured: dict[str, Any] = {}

    async def fake_stream(  # type: ignore[no-untyped-def]
        model, context, api_key, base_url, options, tool_call_id_normalizer=None
    ):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["options"] = options
        captured["tool_call_id_normalizer"] = tool_call_id_normalizer
        return AssistantMessageEventStream()

    with patch("pi_ai.api.mistral_conversations.chat_completions_stream", new=fake_stream):
        await mistral_stream(
            _model(),
            _context(),
            {"api_key": "sk-test", "session_id": "sess-1"},
        )

    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://api.mistral.ai/v1"
    assert captured["options"]["headers"]["x-affinity"] == "sess-1"
    assert captured["tool_call_id_normalizer"] is not None
