"""Proxy 流函数测试（Phase 5.1）。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pi_ai._types import Context, Model

from pi_agent.proxy import process_proxy_event, stream_proxy


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-responses",
        name="Test",
    )


def _partial() -> dict:
    return {
        "role": "assistant",
        "stop_reason": "pending",
        "content": [],
        "api": "openai-responses",
        "provider": "test",
        "model": "test-model",
    }


class TestProcessProxyEvent:
    def test_text_stream_rebuilds_partial(self):
        partial = _partial()
        process_proxy_event({"type": "text_start", "contentIndex": 0}, partial)
        process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "Hello"}, partial)
        process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": " world"}, partial)
        end = process_proxy_event({"type": "text_end", "contentIndex": 0}, partial)

        assert partial["content"][0] == {"type": "text", "text": "Hello world"}
        assert end["content"] == "Hello world"

    def test_toolcall_stream_rebuilds_arguments(self):
        partial = _partial()
        process_proxy_event(
            {"type": "toolcall_start", "contentIndex": 0, "id": "tc-1", "toolName": "search"},
            partial,
        )
        process_proxy_event(
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"q": "py'}, partial
        )
        process_proxy_event(
            {"type": "toolcall_delta", "contentIndex": 0, "delta": 'thon"}'}, partial
        )
        end = process_proxy_event({"type": "toolcall_end", "contentIndex": 0}, partial)

        block = partial["content"][0]
        assert block["type"] == "toolCall"
        assert block["name"] == "search"
        assert block["arguments"] == {"q": "python"}
        assert "partialJson" not in block
        assert end["tool_call"] is block

    def test_done_event_sets_stop_reason_and_usage(self):
        partial = _partial()
        event = process_proxy_event(
            {"type": "done", "reason": "stop", "usage": {"input": 1, "output": 2}},
            partial,
        )
        assert partial["stop_reason"] == "stop"
        assert partial["usage"] == {"input": 1, "output": 2}
        assert event["message"] is partial

    def test_error_event_sets_error_message(self):
        partial = _partial()
        event = process_proxy_event(
            {"type": "error", "reason": "error", "errorMessage": "boom", "usage": {}},
            partial,
        )
        assert partial["stop_reason"] == "error"
        assert partial["error_message"] == "boom"
        assert event["error"] is partial

    def test_unknown_event_ignored(self):
        assert process_proxy_event({"type": "unknown_thing"}, _partial()) is None


class TestStreamProxy:
    @staticmethod
    def _sse_body(events: list[dict]) -> str:
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    @pytest.mark.asyncio
    async def test_stream_proxy_reconstructs_message(self):
        body = self._sse_body(
            [
                {"type": "start"},
                {"type": "text_start", "contentIndex": 0},
                {"type": "text_delta", "contentIndex": 0, "delta": "Hi from proxy"},
                {"type": "text_end", "contentIndex": 0},
                {"type": "done", "reason": "stop", "usage": {"input": 5, "output": 3}},
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer token123"
            assert "/api/stream" in str(request.url)
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        stream = stream_proxy(
            _make_model(),
            Context(system_prompt="", messages=[]),
            {"proxyUrl": "https://proxy.example.com", "authToken": "token123"},
            client=client,
        )

        events: list[dict] = []
        async for event in stream:
            events.append(event)

        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert types[-1] == "done"
        message = await stream.result()
        assert message["stop_reason"] == "stop"
        text = "".join(block["text"] for block in message["content"] if block.get("type") == "text")
        assert text == "Hi from proxy"

    @pytest.mark.asyncio
    async def test_stream_proxy_error_response(self):
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        stream = stream_proxy(
            _make_model(),
            Context(system_prompt="", messages=[]),
            {"proxyUrl": "https://proxy.example.com", "authToken": "bad"},
            client=client,
        )

        events: list[dict] = []
        async for event in stream:
            events.append(event)
        assert events[-1]["type"] == "error"
        assert events[-1]["reason"] == "error"
        assert "unauthorized" in events[-1]["error"]["error_message"]

    @pytest.mark.asyncio
    async def test_stream_proxy_abort(self):
        body = self._sse_body(
            [
                {"type": "text_start", "contentIndex": 0},
                {"type": "text_delta", "contentIndex": 0, "delta": "partial"},
            ]
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        signal = asyncio.Event()
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        stream = stream_proxy(
            _make_model(),
            Context(system_prompt="", messages=[]),
            {"proxyUrl": "https://proxy.example.com", "authToken": "t", "signal": signal},
            client=client,
        )

        signal.set()  # 读取前即中止 → 后台任务抛 aborted 错误事件
        events: list[dict] = []
        async for event in stream:
            events.append(event)
        assert events[-1]["type"] == "error"
        assert events[-1]["reason"] == "aborted"
