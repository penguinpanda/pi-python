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


def test_requires_tool_call_id_matrix() -> None:
    """requiresToolCallId：claude-* / gpt-oss-* / gemini 3+ 需要显式工具调用 ID。"""
    from pi_ai.api.google_generative_ai import _requires_tool_call_id

    assert _requires_tool_call_id("claude-3-7-sonnet") is True
    assert _requires_tool_call_id("gpt-oss-120b") is True
    assert _requires_tool_call_id("gemini-3-pro") is True
    assert _requires_tool_call_id("gemini-2.5-flash") is False
    assert _requires_tool_call_id("gemini-flash") is False


def test_tool_call_ids_emitted_for_gemini3() -> None:
    """Gemini 3+：functionCall/functionResponse 携带归一化 id（对齐 TS convertMessages）。"""
    from pi_ai.api.google_generative_ai import _to_google_contents

    context = Context(
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-1 with space!",
                        "name": "read",
                        "arguments": {"path": "a.txt"},
                    }
                ],
            },
            {
                "role": "toolResult",
                "tool_call_id": "call-1 with space!",
                "tool_name": "read",
                "content": [{"type": "text", "text": "ok"}],
                "is_error": False,
                "timestamp": 1,
            },
        ],
    )
    contents = _to_google_contents(
        context, Model(id="gemini-3-pro", provider="google", api="google-generative-ai")
    )
    assistant_parts = contents[1]["parts"]
    assert assistant_parts[0]["functionCall"]["id"] == "call-1_with_space_"
    function_response = contents[2]["parts"][0]["functionResponse"]
    assert function_response["id"] == "call-1_with_space_"

    # gemini 2.5：不携带 id
    contents_legacy = _to_google_contents(
        context, Model(id="gemini-2.5-flash", provider="google", api="google-generative-ai")
    )
    assert "id" not in contents_legacy[1]["parts"][0]["functionCall"]
    assert "id" not in contents_legacy[2]["parts"][0]["functionResponse"]


@pytest.mark.asyncio
async def test_google_request_retries_on_429(monkeypatch) -> None:
    """429（带 retry-after）在 maxRetries 内指数退避重试后成功（对齐 retryGoogleRequest）。"""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"retry-after-ms": "10"})
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = google_generative_ai.google_generative_ai_stream(
        _model(),
        _context(),
        "sk-test",
        "https://generativelanguage.googleapis.com/v1beta",
        {"maxRetries": 2, "maxRetryDelayMs": 1000},
    )
    events = [event async for event in stream]
    assert events[-1]["type"] == "done"
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_google_request_retries_on_429_snake_case_options(monkeypatch) -> None:
    """snake_case max_retries（StreamOptions 声明键名）同样触发重试。"""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(429, headers={"retry-after-ms": "10"})
        return httpx.Response(200, text=_text_sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = google_generative_ai.google_generative_ai_stream(
        _model(),
        _context(),
        "sk-test",
        "https://generativelanguage.googleapis.com/v1beta",
        {"max_retries": 2, "max_retry_delay_ms": 1000},
    )
    events = [event async for event in stream]
    assert events[-1]["type"] == "done"
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_google_request_no_retry_without_max_retries(monkeypatch) -> None:
    """默认 maxRetries=0：429 不重试，直接 error 事件。"""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429, headers={"retry-after-ms": "10"})

    monkeypatch.setattr(
        google_generative_ai,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = google_generative_ai.google_generative_ai_stream(
        _model(), _context(), "sk-test", "https://generativelanguage.googleapis.com/v1beta"
    )
    events = [event async for event in stream]
    assert events[-1]["type"] == "error"
    assert len(attempts) == 1


def test_thought_signature_echo_and_thought_true() -> None:
    """同 provider/model：thinking 带 thought:true 且回显签名；空块带签名保留。"""
    from pi_ai.api.google_generative_ai import _to_google_contents

    signature = "c2lnbmF0dXJlLXNpZ25hdHVyZQ=="  # "signature-signature" base64
    context = Context(
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "provider": "google",
                "model": "gemini-3-pro",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning here",
                        "thinking_signature": signature,
                    },
                    {"type": "text", "text": "", "text_signature": signature},
                    {"type": "text", "text": "answer"},
                ],
            },
        ],
    )
    contents = _to_google_contents(
        context, Model(id="gemini-3-pro", provider="google", api="google-generative-ai")
    )
    parts = contents[1]["parts"]
    assert parts[0] == {"thought": True, "text": "reasoning here", "thoughtSignature": signature}
    # 空文本块带签名时保留并回显
    assert parts[1] == {"text": "", "thoughtSignature": signature}
    assert parts[2] == {"text": "answer"}

    # 跨 provider/model：签名丢弃，thinking 转纯文本
    context_cross = Context(
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "provider": "openai",
                "model": "gpt-x",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "other reasoning",
                        "thinking_signature": signature,
                    },
                ],
            },
        ],
    )
    contents_cross = _to_google_contents(
        context_cross, Model(id="gemini-3-pro", provider="google", api="google-generative-ai")
    )
    assert contents_cross[1]["parts"] == [{"text": "other reasoning"}]
