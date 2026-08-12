"""Azure OpenAI Responses API 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api import azure_openai_responses
from pi_ai._types import Context, Model
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model() -> Model:
    return Model(
        id="gpt-5-chat-latest", provider="azure-openai-responses", api="azure-openai-responses"
    )


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi"}])


def test_azure_api_registered() -> None:
    assert get_api_provider("azure-openai-responses") is not None


@pytest.mark.asyncio
async def test_azure_stream_builds_client_and_deployment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAzureClient:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setattr(azure_openai_responses, "AsyncAzureOpenAI", _FakeAzureClient)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://res.openai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5-chat-latest=deploy1")

    fake_responses = AsyncMock(return_value=AssistantMessageEventStream())
    with patch("pi_ai.api.azure_openai_responses.responses_stream", new=fake_responses):
        stream = await azure_openai_responses.azure_stream(_model(), _context(), {"api_key": "k"})

    assert stream is not None
    call_kwargs = fake_responses.call_args.kwargs
    assert call_kwargs["request_model_id"] == "deploy1"
    factory = call_kwargs["client_factory"]
    factory("k", "", timeout=1.0, max_retries=2, headers=None)
    assert captured["azure_endpoint"] == "https://res.openai.azure.com"
    assert captured["azure_deployment"] == "deploy1"
    assert captured["api_version"] == "2024-10-21"
