"""retry_provider_request 单元测试。

覆盖取消语义与 HTTP 状态码探测：

- 请求执行中被取消 → CancelledError 立即传播，不重试
- httpx.HTTPStatusError 经 response.status_code 探测（400/401 不重试）
- 429 携带 retry-after-ms → 退避后重试成功
- signal 恰好置位时任务取消仍原样传播（不得转成 _AbortError）
- 退避期间 signal 置位 → 立即中止
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pi_ai.utils.provider_retry import _AbortError, retry_provider_request


class _StatusError(RuntimeError):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status = status
        self.headers = headers or {}


def _http_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/test")
    response = httpx.Response(status, request=request, headers=headers or {})
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.asyncio
async def test_cancel_during_request_propagates_immediately() -> None:
    """task.cancel() 产生的 CancelledError 必须立即传播，不得进入重试退避。"""
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_provider_request(request, max_retries=3)
    assert calls == 1


@pytest.mark.asyncio
async def test_http_400_not_retried() -> None:
    """HTTPStatusError 400 经 response.status_code 探测为不可重试。"""
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise _http_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_provider_request(request, max_retries=3)
    assert calls == 1


@pytest.mark.asyncio
async def test_http_429_retried_then_succeeds() -> None:
    """429 携带 retry-after-ms=1 → 退避后重试成功。"""
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(429, {"retry-after-ms": "1"})
        return "ok"

    result = await retry_provider_request(request, max_retries=3)
    assert result == "ok"
    assert calls == 2


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
