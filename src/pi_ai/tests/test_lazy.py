"""懒加载机制（lazy_stream / lazy_api / forward_stream）测试。"""

import asyncio

import pytest

from pi_ai.api.lazy import forward_stream, lazy_api, lazy_stream
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    Usage,
    now_ms,
)
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="test-api",
    )


def _context() -> Context:
    return Context(messages=[], tools=[])


def _done_message() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api="test-api",
        provider="test",
        model="test-model",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        ),
        stop_reason="stop",
        timestamp=now_ms(),
    )


async def _inner_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = _done_message()
    stream.push({"type": "done", "reason": "stop", "message": message})
    return stream


@pytest.mark.asyncio
async def test_lazy_stream_forwards_events():
    called = asyncio.Event()

    async def setup():
        called.set()
        return await _inner_stream()

    outer = lazy_stream(_model(), setup)
    events = [event async for event in outer]
    assert [event["type"] for event in events] == ["done"]
    assert called.is_set()
    result = await outer.result()
    assert result["stop_reason"] == "stop"


@pytest.mark.asyncio
async def test_lazy_stream_setup_failure_degrades_to_error_event():
    async def setup():
        raise RuntimeError("auth failed")

    outer = lazy_stream(_model(), setup)
    events = [event async for event in outer]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["reason"] == "error"
    assert "auth failed" in events[0]["error"]["error_message"]
    result = await outer.result()
    assert result["stop_reason"] == "error"
    assert "auth failed" in result["error_message"]


@pytest.mark.asyncio
async def test_forward_stream_ends_target_with_source_result():
    target = AssistantMessageEventStream()
    source = await _inner_stream()
    await forward_stream(target, source)
    events = [event async for event in target]
    assert [event["type"] for event in events] == ["done"]
    assert (await target.result())["stop_reason"] == "stop"


@pytest.mark.asyncio
async def test_lazy_api_loads_module_on_first_call():
    loads = 0

    async def load():
        nonlocal loads
        loads += 1
        return _FakeStreams()

    api = lazy_api(load)
    outer1 = api.stream(_model(), _context())
    events1 = [event async for event in outer1]
    assert events1[0]["type"] == "done"
    outer2 = api.stream(_model(), _context())
    await outer2.result()
    # 调用方负责去重（TS 依赖 import 缓存；此处 load 被调用次数由实现决定）。
    assert loads >= 1


@pytest.mark.asyncio
async def test_lazy_api_failure_becomes_error_event():
    async def load():
        raise ImportError("module missing")

    api = lazy_api(load)
    outer = api.stream(_model(), _context())
    events = [event async for event in outer]
    assert events[0]["type"] == "error"


class _FakeStreams:
    def stream(self, model, context, options=None):
        return _make_stream()

    def streamSimple(self, model, context, options=None):
        return _make_stream()


def _make_stream() -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = _done_message()
    stream.push({"type": "done", "reason": "stop", "message": message})
    return stream
