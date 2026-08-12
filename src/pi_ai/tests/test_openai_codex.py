"""OpenAI Codex Responses API/provider 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pi_ai import Models, create_default_models
from pi_ai.api import openai_codex_responses
from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.openai_codex_responses import codex_cancel_deferred, codex_fetch_deferred
from pi_ai._types import Context, Model
from pi_ai.providers.openai_codex import openai_codex_provider
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model() -> Model:
    return Model(id="gpt-5.4", provider="openai-codex", api="openai-codex-responses")


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi"}])


def test_openai_codex_api_and_provider_registered() -> None:
    assert get_api_provider("openai-codex-responses") is not None
    assert create_default_models().get_provider("openai-codex") is not None


def test_openai_codex_provider_has_oauth() -> None:
    provider = openai_codex_provider()
    assert getattr(provider.auth, "oauth", None) is not None
    assert provider.get_models()


def test_openai_codex_models_enable_tool_search() -> None:
    provider = openai_codex_provider()
    assert all((model.compat or {}).get("supportsToolSearch") for model in provider.get_models())


def test_openai_codex_provider_supports_deferred() -> None:
    models = Models()
    models.add_provider(openai_codex_provider())
    model = models.get_model("openai-codex", "gpt-5.4")
    assert model is not None
    assert models.supports_deferred(model) is True


@pytest.mark.asyncio
async def test_codex_stream_builds_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setattr(openai_codex_responses, "AsyncOpenAI", _FakeClient)
    fake_responses = AsyncMock(return_value=AssistantMessageEventStream())
    with patch("pi_ai.api.openai_codex_responses.responses_stream", new=fake_responses):
        stream = await openai_codex_responses.codex_stream(
            _model(), _context(), {"api_key": "k", "chatgpt_account_id": "u-1"}
        )

    assert stream is not None
    call_kwargs = fake_responses.call_args.kwargs
    assert call_kwargs["request_model_id"] == "gpt-5.4"
    factory = call_kwargs["client_factory"]
    factory("k", "", timeout=1.0, max_retries=2, headers=None)
    assert captured["base_url"] == "https://chatgpt.com/backend-api"
    assert captured["default_headers"]["OpenAI-Beta"] == "responses=experimental"
    assert captured["default_headers"]["chatgpt-account-id"] == "u-1"


@pytest.mark.asyncio
async def test_codex_fetch_deferred(monkeypatch) -> None:
    class _TextBlock:
        type = "output_text"
        text = "done"

    class _MessageItem:
        type = "message"
        content = [_TextBlock()]

    class _Usage:
        input_tokens = 3
        output_tokens = 2

    class _Response:
        id = "resp-1"
        status = "completed"
        usage = _Usage()
        output = [_MessageItem()]

    class _Responses:
        async def retrieve(self, response_id: str) -> _Response:
            assert response_id == "deferred-1"
            return _Response()

    class _FakeClient:
        responses = _Responses()

    monkeypatch.setattr(openai_codex_responses, "_codex_client", lambda *a, **k: _FakeClient())
    message = await codex_fetch_deferred(
        _model(),
        {"id": "deferred-1"},
        {"api_key": "k"},
    )
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "done"
    assert message["stop_reason"] == "stop"
    assert message["usage"]["input"] == 3


@pytest.mark.asyncio
async def test_codex_cancel_deferred(monkeypatch) -> None:
    cancelled: list[str] = []

    class _Responses:
        async def cancel(self, response_id: str) -> None:
            cancelled.append(response_id)

    class _FakeClient:
        responses = _Responses()

    monkeypatch.setattr(openai_codex_responses, "_codex_client", lambda *a, **k: _FakeClient())
    await codex_cancel_deferred(_model(), {"id": "deferred-2"}, {"api_key": "k"})
    assert cancelled == ["deferred-2"]
