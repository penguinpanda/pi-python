"""Google Generative AI API 测试。"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai.api import google_generative_ai
from pi_ai.api.google_generative_ai import _supports_google_strict_tool_sampling
from pi_ai._types import Context, Model, Tool


def _model() -> Model:
    return Model(id="gemini-2.5-flash", provider="google", api="google-generative-ai")


def _context() -> Context:
    return Context(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[Tool(name="read", description="Read a file", input_schema={"type": "object"})],
    )


def _text_sse() -> str:
    return (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":0,'
        '"candidatesTokenCount":2,"totalTokenCount":12}}\n\n'
    )


def _tool_sse() -> str:
    return (
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read",'
        '"args":{"path":"a.txt"}}}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":4,"cachedContentTokenCount":0,'
        '"candidatesTokenCount":3,"totalTokenCount":7}}\n\n'
    )


@pytest.mark.asyncio
async def test_google_text_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = google_generative_ai.google_generative_ai_stream(
        _model(), _context(), "sk-test", "https://generativelanguage.googleapis.com/v1beta"
    )
    events = [event async for event in stream]
    message = events[-1]["message"]

    assert captured["headers"]["x-goog-api-key"] == "sk-test"
    assert captured["payload"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert captured["payload"]["tools"][0]["functionDeclarations"][0]["name"] == "read"
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Hello"
    assert message["stop_reason"] == "stop"
    assert message["usage"]["input"] == 10


@pytest.mark.asyncio
async def test_google_tool_stream(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_tool_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = google_generative_ai.google_generative_ai_stream(
        _model(), _context(), "sk-test", "https://generativelanguage.googleapis.com/v1beta"
    )
    events = [event async for event in stream]
    message = events[-1]["message"]
    tool = message["content"][0]

    assert tool["type"] == "toolCall"
    assert tool["name"] == "read"
    assert tool["arguments"] == {"path": "a.txt"}
    assert message["stop_reason"] == "tool_call"


def test_google_strict_tool_sampling_detection() -> None:
    assert _supports_google_strict_tool_sampling("gemini-2.5-flash") is False
    assert _supports_google_strict_tool_sampling("gemini-3-pro") is True
    assert _supports_google_strict_tool_sampling("gemini-3.1-flash") is True


@pytest.mark.asyncio
async def test_google_strict_tool_validated_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = Model(id="gemini-3-pro", provider="google", api="google-generative-ai", reasoning=True)
    context = Context(
        messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        tools=[
            Tool(
                name="read",
                description="Read a file",
                input_schema={"type": "object"},
                constrained_sampling={"type": "json_schema", "strict": "prefer"},
            )
        ],
    )
    stream = google_generative_ai.google_generative_ai_stream(
        model, context, "sk-test", "https://generativelanguage.googleapis.com/v1beta"
    )
    _ = [event async for event in stream]

    payload = captured["payload"]
    assert payload["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}


@pytest.mark.asyncio
async def test_google_advanced_thinking_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = Model(id="gemini-3-pro", provider="google", api="google-generative-ai", reasoning=True)
    context = Context(messages=[{"role": "user", "content": "hi", "timestamp": 0}])
    stream = google_generative_ai.google_generative_ai_stream(
        model,
        context,
        "sk-test",
        "https://generativelanguage.googleapis.com/v1beta",
        {"reasoning": "high"},
    )
    _ = [event async for event in stream]

    thinking_config = captured["payload"]["generationConfig"]["thinkingConfig"]
    assert thinking_config["includeThoughts"] is True
    assert thinking_config["thinkingLevel"] == "HIGH"


@pytest.mark.asyncio
async def test_google_thinking_budget(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = Model(
        id="gemini-2.5-pro", provider="google", api="google-generative-ai", reasoning=True
    )
    context = Context(messages=[{"role": "user", "content": "hi", "timestamp": 0}])
    stream = google_generative_ai.google_generative_ai_stream(
        model,
        context,
        "sk-test",
        "https://generativelanguage.googleapis.com/v1beta",
        {"reasoning": "medium"},
    )
    _ = [event async for event in stream]

    thinking_config = captured["payload"]["generationConfig"]["thinkingConfig"]
    assert thinking_config["includeThoughts"] is True
    assert thinking_config["thinkingBudget"] == 8192
