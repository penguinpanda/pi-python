"""OpenAI Codex Responses API/provider 测试。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import zstandard

from pi_ai import create_default_models
from pi_ai.api import openai_codex_responses
from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.openai_codex_responses import (
    _CodexWebSocketClient,
    _CodexWebSocketEvents,
    _CodexZstdTransport,
    codex_cancel_deferred,
    codex_fetch_deferred,
    _resolve_codex_websocket_url,
)
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
    # 模型由 create_default_models() 统一合并生成目录。
    assert create_default_models().get_provider("openai-codex").get_models()


def test_openai_codex_models_enable_tool_search() -> None:
    provider = create_default_models().get_provider("openai-codex")
    assert all((model.compat or {}).get("supportsToolSearch") for model in provider.get_models())


def test_openai_codex_provider_supports_deferred() -> None:
    models = create_default_models()
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
    assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert captured["default_headers"]["OpenAI-Beta"] == "responses=experimental"
    assert captured["default_headers"]["content-encoding"] == "zstd"
    assert captured["default_headers"]["chatgpt-account-id"] == "u-1"


@pytest.mark.asyncio
async def test_codex_zstd_transport_compresses_request_body() -> None:
    captured: dict[str, bytes] = {}

    class _Inner(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured["content"] = request.content
            return httpx.Response(200)

    request = httpx.Request(
        "POST",
        "https://example.com/codex/responses",
        headers={"content-encoding": "zstd"},
        content=b'{"model":"gpt-5.4"}',
    )
    await _CodexZstdTransport(_Inner()).handle_async_request(request)

    decompressed = zstandard.ZstdDecompressor().decompress(captured["content"])
    assert decompressed == b'{"model":"gpt-5.4"}'


def test_codex_websocket_url() -> None:
    assert (
        _resolve_codex_websocket_url("https://chatgpt.com/backend-api")
        == "wss://chatgpt.com/backend-api/codex/responses"
    )


@pytest.mark.asyncio
async def test_codex_websocket_event_stream(monkeypatch) -> None:
    sent: list[str] = []
    closed: list[bool] = []

    class _FakeConn:
        def __init__(self) -> None:
            self._messages = iter(
                [
                    {"type": "response.output_text.delta", "delta": "hi"},
                    {
                        "type": "response.completed",
                        "response": {
                            "output_text": "hi",
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    },
                ]
            )

        async def send(self, data: str) -> None:
            sent.append(data)

        async def recv(self) -> str:
            return json.dumps(next(self._messages))

        async def close(self) -> None:
            closed.append(True)

    async def fake_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeConn()

    monkeypatch.setattr("websockets.connect", fake_connect)
    events = _CodexWebSocketEvents(
        url="wss://example.com/codex/responses",
        headers={"Authorization": "Bearer k"},
        body={"model": "gpt-5.4"},
        options={},
    )
    first = await events.__anext__()
    second = await events.__anext__()

    assert first.type == "response.output_text.delta"
    assert first.delta == "hi"
    assert second.type == "response.completed"
    assert closed == [True]
    sent_payload = json.loads(sent[0])
    assert sent_payload["type"] == "response.create"
    assert sent_payload["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_codex_stream_uses_websocket_factory(monkeypatch) -> None:
    class _FakeConn:
        async def send(self, data: str) -> None:
            pass

        async def recv(self) -> str:
            await asyncio.sleep(3600)
            return ""

        async def close(self) -> None:
            pass

    async def fake_connect(*args, **kwargs):
        return _FakeConn()

    monkeypatch.setattr("websockets.connect", fake_connect)
    fake_responses = AsyncMock(return_value=AssistantMessageEventStream())
    with patch("pi_ai.api.openai_codex_responses.responses_stream", new=fake_responses):
        await openai_codex_responses.codex_stream(
            _model(),
            _context(),
            {"api_key": "k", "transport": "websocket"},
        )

    factory = fake_responses.call_args.kwargs["client_factory"]
    client = factory("k", "", timeout=1.0, max_retries=0, headers=None)
    assert isinstance(client, _CodexWebSocketClient)


@pytest.mark.asyncio
async def test_codex_websocket_stream_integration(monkeypatch) -> None:
    class _FakeConn:
        def __init__(self) -> None:
            self._messages = iter(
                [
                    {"type": "response.output_text.delta", "delta": "Hello"},
                    {
                        "type": "response.completed",
                        "response": {
                            "output_text": "Hello",
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    },
                ]
            )

        async def send(self, data: str) -> None:
            pass

        async def recv(self) -> str:
            return json.dumps(next(self._messages))

        async def close(self) -> None:
            pass

    async def fake_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeConn()

    monkeypatch.setattr("websockets.connect", fake_connect)
    stream = await openai_codex_responses.openai_codex_responses_stream(
        _model(),
        _context(),
        "k",
        "",
        {"transport": "websocket", "api_key": "k"},
    )
    events = [event async for event in stream]
    message = events[-1]["message"]
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Hello"
    assert message["stop_reason"] == "stop"


@pytest.mark.asyncio
async def test_websocket_cached_reuses_connection_with_delta(monkeypatch) -> None:
    """websocket-cached 同 session 应复用连接并发送 previous_response_id + input delta。"""
    from pi_ai.api.openai_codex_responses import _codex_websocket_cache

    _codex_websocket_cache.clear()
    sent: list[dict] = []

    class _FakeConn:
        def __init__(self) -> None:
            self._queue: asyncio.Queue[dict] = asyncio.Queue()

        async def send(self, data: str) -> None:
            sent.append(json.loads(data))

        async def recv(self) -> str:
            return json.dumps(await self._queue.get())

        async def close(self) -> None:
            pass

    conn = _FakeConn()
    output1 = [
        {
            "type": "message",
            "id": "msg1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
        }
    ]
    for event in (
        {"type": "response.output_text.delta", "delta": "Hello"},
        {
            "type": "response.completed",
            "response": {"id": "resp-1", "output_text": "Hello", "output": output1, "usage": None},
        },
        {"type": "response.output_text.delta", "delta": "Again"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-2",
                "output_text": "Again",
                "output": [],
                "usage": None,
            },
        },
    ):
        conn._queue.put_nowait(event)

    async def fake_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return conn

    monkeypatch.setattr("websockets.connect", fake_connect)
    options = {"transport": "websocket-cached", "api_key": "k", "session_id": "cache-session"}
    first = await openai_codex_responses.openai_codex_responses_stream(
        _model(), _context(), "k", "", options
    )
    msg1 = await first.collect()
    assert msg1["content"][0]["text"] == "Hello"

    context2 = Context(
        messages=[
            *_context().messages,
            {
                "role": "assistant",
                "api": "openai-codex-responses",
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "content": [
                    {"type": "text", "text": "Hello", "text_signature": '{"v":1,"id":"msg1"}'}
                ],
                "timestamp": 0,
            },
            {"role": "user", "content": "again", "timestamp": 0},
        ]
    )
    second = await openai_codex_responses.openai_codex_responses_stream(
        _model(), context2, "k", "", options
    )
    await second.collect()
    assert len(sent) == 2
    assert sent[1]["previous_response_id"] == "resp-1"
    assert sent[1]["input"] == [{"role": "user", "content": "again"}]
    _codex_websocket_cache.clear()


@pytest.mark.asyncio
async def test_websocket_connect_failure_falls_back_to_sse(monkeypatch) -> None:
    """WS 连接阶段失败回退 SSE；同会话后续请求直接 SSE（对齐 TS fallback 会话记忆）。"""
    from pi_ai.api.openai_codex_responses import (
        _is_ws_sse_fallback_active,
    )

    async def fake_connect(*args, **kwargs):
        raise ConnectionError("ws refused")

    monkeypatch.setattr("websockets.connect", fake_connect)

    calls: list[str] = []

    async def fake_responses_stream(model, context, api_key, base_url, options=None, **kwargs):
        calls.append("sse")
        stream = AssistantMessageEventStream()
        output = {
            "content": [{"type": "text", "text": "from-sse"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
        }
        stream.push({"type": "start", "partial": output})
        stream.push({"type": "done", "reason": "stop", "message": output})
        stream.end()
        return stream

    monkeypatch.setattr(
        "pi_ai.api.openai_codex_responses.responses_stream",
        fake_responses_stream,
    )
    session_id = "codex-session-fallback-test"
    stream = await openai_codex_responses.openai_codex_responses_stream(
        _model(),
        _context(),
        "k",
        "",
        {"transport": "websocket", "api_key": "k", "session_id": session_id},
    )
    events = [event async for event in stream]
    assert calls == ["sse"]
    assert events[-1]["message"]["content"][0]["text"] == "from-sse"
    assert _is_ws_sse_fallback_active(session_id)
    # 同会话第二次请求：回退记忆生效，不再尝试 WS。
    await openai_codex_responses.openai_codex_responses_stream(
        _model(),
        _context(),
        "k",
        "",
        {"transport": "websocket", "api_key": "k", "session_id": session_id},
    )
    assert calls == ["sse", "sse"]


@pytest.mark.asyncio
async def test_websocket_send_failure_falls_back_to_sse(monkeypatch) -> None:
    """WS 连接成功但 response.create 发送失败：未产生事件，应回退 SSE。"""
    from pi_ai.api.openai_codex_responses import _is_ws_sse_fallback_active

    class _FakeConn:
        async def send(self, data: str) -> None:
            raise ConnectionError("send refused")

        async def close(self) -> None:
            pass

    async def fake_connect(*args, **kwargs):
        return _FakeConn()

    monkeypatch.setattr("websockets.connect", fake_connect)
    calls: list[str] = []

    async def fake_responses_stream(model, context, api_key, base_url, options=None, **kwargs):
        calls.append("sse")
        stream = AssistantMessageEventStream()
        output = {
            "content": [{"type": "text", "text": "from-sse"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "stop_reason": "stop",
        }
        stream.push({"type": "start", "partial": output})
        stream.push({"type": "done", "reason": "stop", "message": output})
        stream.end()
        return stream

    monkeypatch.setattr(
        "pi_ai.api.openai_codex_responses.responses_stream",
        fake_responses_stream,
    )
    stream = await openai_codex_responses.openai_codex_responses_stream(
        _model(),
        _context(),
        "k",
        "",
        {"transport": "websocket", "api_key": "k", "session_id": "codex-send-fail-test"},
    )
    events = [event async for event in stream]
    assert calls == ["sse"]
    assert events[-1]["message"]["content"][0]["text"] == "from-sse"
    assert _is_ws_sse_fallback_active("codex-send-fail-test")


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
