"""pi-messages 线协议测试（mock HTTP）。"""

import json

import httpx
import pytest

from pi_ai.api.pi_messages import (
    PiMessagesResponseError,
    read_pi_messages_events,
    stream,
)
from pi_ai.api.pi_messages_lazy import pi_messages_api
from pi_ai.types import Context, Model


def _model() -> Model:
    return Model(
        id="router-model",
        provider="radius",
        api="pi-messages",
        base_url="https://gateway.test",
    )


def _context() -> Context:
    return Context(
        messages=[{"role": "user", "content": "hi", "timestamp": 1}],
        system_prompt="You are helpful",
    )


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


@pytest.mark.asyncio
async def test_read_events_chunking_and_crlf():
    raw = b'data: {"type": "start"}\r\n\r\ndata: {"type": "text_delta", "contentIndex": 0'
    raw += b', "delta": "x"}\n\ndata: [DONE]\n\n'
    collected = []
    async for event in read_pi_messages_events(_byte_chunks(raw, 7)):
        collected.append(event)
    assert [e["type"] for e in collected] == ["start", "text_delta"]


async def _byte_chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_stream_full_sequence(monkeypatch):
    events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
        {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
        {"type": "text_end", "contentIndex": 0, "content": "Hello"},
        {
            "type": "done",
            "reason": "stop",
            "usage": {
                "input": 5,
                "output": 1,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 6,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/messages"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "router-model"
        assert body["options"]["sessionId"] == "s-1"
        return httpx.Response(
            200, content=_sse(*events), headers={"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(
        "pi_ai.api.pi_messages._AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = stream(
        _model(),
        _context(),
        {"api_key": "sk-test", "session_id": "s-1"},
    )
    collected = [event async for event in result]
    assert [e["type"] for e in collected] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    message = await result.result()
    assert message["stop_reason"] == "stop"
    assert message["content"][0]["text"] == "Hello"
    assert message["usage"]["input"] == 5


@pytest.mark.asyncio
async def test_stream_toolcall_streaming_json(monkeypatch):
    events = [
        {"type": "toolcall_start", "contentIndex": 0, "id": "call_1", "toolName": "search"},
        {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"q'},
        {"type": "toolcall_delta", "contentIndex": 0, "delta": 'uery": "x"}'},
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {
                "id": "call_1",
                "name": "search",
                "arguments": {"query": "x"},
                "raw_arguments": '{"query": "x"}',
            },
        },
        {"type": "done", "reason": "toolUse", "usage": None},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(*events))

    monkeypatch.setattr(
        "pi_ai.api.pi_messages._AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = stream(_model(), _context(), {"api_key": "k"})
    toolcall_events = [
        e for e in [event async for event in result] if e["type"].startswith("toolcall")
    ]
    assert [e["type"] for e in toolcall_events] == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    # delta 增量累积（partial 为共享可变引用，终态断言看 result）
    assert toolcall_events[1]["delta"] == '{"q'
    assert toolcall_events[2]["delta"] == 'uery": "x"}'
    assert toolcall_events[3]["tool_call"]["arguments"] == {"query": "x"}
    message = await result.result()
    assert message["stop_reason"] == "tool_call"
    assert message["content"][0]["arguments"] == {"query": "x"}


@pytest.mark.asyncio
async def test_stream_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "gateway exploded", "code": "E_500"}},
        )

    monkeypatch.setattr(
        "pi_ai.api.pi_messages._AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = stream(_model(), _context(), {"api_key": "k"})
    events = [event async for event in result]
    assert events[0]["type"] == "error"
    assert events[0]["reason"] == "error"
    assert "gateway exploded" in events[0]["error"]["error_message"]
    # 诊断信息（PiMessagesResponseError）
    diagnostics = events[0]["error"].get("diagnostics") or []
    assert any(d["type"] == "pi_messages_response_failure" for d in diagnostics)


@pytest.mark.asyncio
async def test_stream_no_terminal_event(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"type": "start"}))

    monkeypatch.setattr(
        "pi_ai.api.pi_messages._AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = stream(_model(), _context(), {"api_key": "k"})
    events = [event async for event in result]
    assert events[-1]["type"] == "error"
    assert "terminal event" in events[-1]["error"]["error_message"]


@pytest.mark.asyncio
async def test_pi_messages_api_lazy(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"type": "done", "reason": "stop", "usage": None}))

    monkeypatch.setattr(
        "pi_ai.api.pi_messages._AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    api = pi_messages_api()
    result = api.stream(_model(), _context(), {"api_key": "k"})
    events = [event async for event in result]
    assert events[0]["type"] == "done"


def test_response_error_holds_code():
    err = PiMessagesResponseError("500 Internal Server Error: boom", "E_500", {"x": 1})
    assert err.code == "E_500"
    assert err.diagnostic_details == {"x": 1}
