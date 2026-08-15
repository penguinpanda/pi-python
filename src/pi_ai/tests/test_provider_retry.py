"""provider_retry（可中断重试）单元测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_ai.utils.provider_retry import _AbortError, retry_provider_request


class _StatusError(RuntimeError):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status = status
        self.headers = headers or {}


@pytest.mark.asyncio
async def test_cancelled_error_not_swallowed_when_signal_set():
    """signal 恰好置位时，任务取消也必须原样传播（不得转成 _AbortError）。"""
    signal = asyncio.Event()
    signal.set()

    async def request():
        await asyncio.sleep(3600)

    task = asyncio.create_task(retry_provider_request(request, max_retries=3, signal=signal))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_signal_aborts_during_backoff():
    signal = asyncio.Event()

    async def request():
        raise _StatusError(503, {"retry-after": "5"})

    async def set_signal():
        await asyncio.sleep(0.02)
        signal.set()

    setter = asyncio.create_task(set_signal())
    with pytest.raises(_AbortError):
        await retry_provider_request(request, max_retries=2, signal=signal)
    await setter


@pytest.mark.asyncio
async def test_retry_then_success():
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _StatusError(429)
        return "ok"

    result = await retry_provider_request(request, max_retries=2)
    assert result == "ok"
    assert attempts == 2
