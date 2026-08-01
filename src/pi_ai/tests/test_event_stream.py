"""
Unit tests for EventStream.
"""

import asyncio

import pytest

from pi_ai._event_stream import AssistantMessageEventStream, EventStream
from pi_ai._types import (
    AssistantMessage,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
)


class TestEventStream:
    """Generic EventStream tests."""

    @pytest.mark.asyncio
    async def test_push_and_iterate(self):
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e,
        )

        stream.push(1)
        stream.push(2)
        stream.push(-1)  # terminal

        results = []
        async for event in stream:
            results.append(event)

        assert results == [1, 2, -1]

    @pytest.mark.asyncio
    async def test_result_await(self):
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == 99,
            extract_result=lambda e: e * 10,
        )

        # Push events in background
        async def produce():
            await asyncio.sleep(0.01)
            stream.push(1)
            stream.push(99)

        asyncio.ensure_future(produce())

        result = await stream.result()
        assert result == 990

    @pytest.mark.asyncio
    async def test_end_method(self):
        stream: EventStream[str, str] = EventStream(
            is_complete=lambda e: e == "done",
            extract_result=lambda e: e,
        )

        stream.push("hello")
        stream.end("final")

        events = [e async for e in stream]
        assert events == ["hello"]
        assert await stream.result() == "final"

    @pytest.mark.asyncio
    async def test_error_method(self):
        """error() ends the stream and makes result() raise."""
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e,
        )

        stream.push(1)
        stream.error(ValueError("boom"))

        # async for stops cleanly after error()
        events = [e async for e in stream]
        assert events == [1]

        # result() raises instead of hanging forever
        with pytest.raises(ValueError, match="boom"):
            await stream.result()

    @pytest.mark.asyncio
    async def test_end_without_result_completes(self):
        """end() without a result must still complete result() (no hang)."""
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e,
        )
        stream.push(1)
        stream.end()

        assert [e async for e in stream] == [1]
        assert await stream.result() is None


class TestAssistantMessageEventStream:
    """AssistantMessageEventStream specialized tests."""

    @pytest.mark.asyncio
    async def test_done_event(self):
        stream = AssistantMessageEventStream()
        msg = AssistantMessage(
            role="assistant",
            content=[{"type": "text", "text": "Hello"}],
            api="openai-completions",
            provider="deepseek",
            model="deepseek-chat",
            usage={"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15},
            stopReason="end",
            errorMessage=None,
            timestamp=0,
        )

        stream.push({"type": "delta", "text": "Hello"})
        stream.push({"type": "done", "message": msg})

        events = [e async for e in stream]
        assert len(events) == 2
        assert events[0]["type"] == "delta"
        assert events[1]["type"] == "done"

        result = await stream.result()
        assert result["stopReason"] == "end"

    @pytest.mark.asyncio
    async def test_error_event(self):
        stream = AssistantMessageEventStream()
        err_msg = AssistantMessage(
            role="assistant",
            content=[],
            api="openai-completions",
            provider="deepseek",
            model="deepseek-chat",
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="error",
            errorMessage="Something went wrong",
            timestamp=0,
        )

        stream.push({"type": "error", "reason": "auth", "error": err_msg})

        events = [e async for e in stream]
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["error"]["errorMessage"] == "Something went wrong"


class TestEventStreamCancellation:
    """Stream cancellation and edge cases."""

    @pytest.mark.asyncio
    async def test_cancellation_no_deadlock(self):
        """Cancelling async iteration should not cause deadlock."""
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e,
        )

        stream.push(1)
        stream.push(2)

        events = []
        try:
            async for event in stream:
                events.append(event)
                if event == 2:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            pass

        # Should have collected events up to cancellation point
        assert events == [1, 2]

    @pytest.mark.asyncio
    async def test_end_before_iteration(self):
        """Calling end() before any push should produce empty stream."""
        stream: EventStream[str, str] = EventStream(
            is_complete=lambda e: e == "done",
            extract_result=lambda e: e,
        )
        stream.end("result")

        events = [e async for e in stream]
        assert events == []
        assert await stream.result() == "result"

    @pytest.mark.asyncio
    async def test_push_after_end_is_noop(self):
        """Events pushed after end() should not appear in iteration."""
        stream: EventStream[int, int] = EventStream(
            is_complete=lambda e: e == -1,
            extract_result=lambda e: e,
        )

        stream.push(1)
        stream.end(42)
        stream.push(2)  # should be a no-op

        events = [e async for e in stream]
        assert events == [1]
