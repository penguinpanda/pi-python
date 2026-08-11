"""多 API Provider 按 model.api 分发测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pi_ai._types import AssistantMessage, Context, Model
from pi_ai.api.api_provider_registry import (
    ApiProvider,
    register_api_provider,
    reset_api_providers,
)
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.provider import create_provider
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model(model_id: str, api: str) -> Model:
    return Model(id=model_id, provider="multi", api=api)


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "Hi"}])  # type: ignore[typeddict-unknown-key]


def _done_message(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total_tokens": 0},
        stop_reason="stop",
        error_message=None,
        timestamp=0,
    )


def _stub_provider(api: str, record: list) -> ApiProvider:
    def _stream(model: Model, context: Context, options=None):  # type: ignore[no-untyped-def]
        record.append(model.id)
        stream = AssistantMessageEventStream()
        stream.push({"type": "done", "reason": "stop", "message": _done_message(model)})
        stream.end()
        return stream

    return ApiProvider(api=api, stream=_stream, streamSimple=_stream)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_api_providers()
    yield
    reset_api_providers()


@pytest.mark.asyncio
async def test_provider_dispatches_by_model_api() -> None:
    anthropic_record: list[str] = []
    responses_record: list[str] = []
    register_api_provider(_stub_provider("anthropic-messages", anthropic_record), source_id="test")
    register_api_provider(_stub_provider("openai-responses", responses_record), source_id="test")
    provider = create_provider(
        id="multi",
        name="Multi API",
        auth=EnvApiKeyAuth("Multi", ["MULTI_API_KEY"]),
        models=[_model("a", ""), _model("b", "openai-responses")],
        api_kind="anthropic-messages",
    )

    with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
        await provider.stream(_model("a", ""), _context())
        await provider.stream(_model("b", "openai-responses"), _context())

    assert anthropic_record == ["a"]
    assert responses_record == ["b"]
