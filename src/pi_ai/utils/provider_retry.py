"""Provider 请求重试（对齐 TS utils/provider-retry.ts）。

复刻 OpenAI/Anthropic SDK 的重试策略（408/409/429/5xx + 指数退避，
尊重 retry-after / x-should-retry 头），但退避可被取消信号中断。
调用方应把 SDK 自身重试关掉（maxRetries=0）后用本助手包装请求。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

DEFAULT_MAX_RETRY_DELAY_MS = 60_000

_ProviderRequest = Callable[[], Awaitable[Any]]


class _AbortError(Exception):
    pass


def _error_status(error: BaseException) -> int | None:
    status = getattr(error, "status", None)
    if status is None:
        status = getattr(error, "status_code", None)
    if status is None:
        # httpx.HTTPStatusError 不带顶层 status_code，经 response 间接携带。
        response = getattr(error, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _error_headers(error: BaseException) -> dict | None:
    headers = getattr(error, "headers", None)
    if headers is None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    if isinstance(headers, dict):
        return headers
    return dict(headers)


def _is_provider_error(error: BaseException) -> bool:
    # 对齐 TS isProviderError：需携带 status 与 headers 信息
    # （值可为 None；httpx 异常经 response 间接携带）。
    has_status = (
        hasattr(error, "status") or hasattr(error, "status_code") or hasattr(error, "response")
    )
    return has_status and _error_headers(error) is not None


def _is_retryable_provider_error(error: BaseException) -> bool:
    headers = _error_headers(error)
    if headers is not None:
        should_retry = headers.get("x-should-retry")
        if should_retry == "true":
            return True
        if should_retry == "false":
            return False
    status = _error_status(error)
    if status is None:
        return True
    return status in (408, 409, 429) or status >= 500


def _validate_server_retry_delay_ms(
    delay_ms: float, max_retry_delay_ms: int | None, provider_message: str
) -> float:
    max_delay = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay > 0 and delay_ms > max_delay:
        raise RuntimeError(
            f"Server requested {delay_ms / 1000:.0f}s retry delay "
            f"(max: {max_delay / 1000:.0f}s). {provider_message}"
        )
    return delay_ms


def _get_retry_delay_ms(
    error: BaseException, retry_index: int, max_retry_delay_ms: int | None
) -> float:
    headers = _error_headers(error) or {}
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_retry_delay_ms(
                float(retry_after_ms), max_retry_delay_ms, str(error)
            )
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return _validate_server_retry_delay_ms(
                float(retry_after) * 1000, max_retry_delay_ms, str(error)
            )
        except ValueError:
            pass

    exponential_delay = min(0.5 * (2**retry_index), 8) * 1000
    return exponential_delay


async def _abortable_sleep(delay_ms: float, signal: Any) -> None:
    """可中断退避：signal（asyncio.Event）置位时立即抛 _AbortError。"""
    if signal is not None and signal.is_set():
        raise _AbortError("Request aborted")
    if delay_ms <= 0:
        return
    if signal is None:
        await asyncio.sleep(delay_ms / 1000)
        return
    try:
        await asyncio.wait_for(signal.wait(), timeout=delay_ms / 1000)
    except asyncio.TimeoutError:
        return
    if signal.is_set():
        raise _AbortError("Request aborted")


async def retry_provider_request(
    request: _ProviderRequest,
    *,
    max_retries: int = 0,
    max_retry_delay_ms: int | None = None,
    signal: Any = None,
) -> Any:
    """带重试的 provider 请求（对齐 TS retryProviderRequest）。"""
    retries_remaining = max_retries
    while True:
        try:
            return await request()
        except _AbortError:
            raise
        except asyncio.CancelledError:
            # 外部 task.cancel() 必须立即传播，不得被当作可重试错误吞掉重试。
            raise
        except Exception as error:
            if signal is not None and signal.is_set():
                raise _AbortError("Request aborted") from None
            if retries_remaining <= 0 or not _is_retryable_provider_error(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            delay_ms = _get_retry_delay_ms(error, retry_index, max_retry_delay_ms)
            await _abortable_sleep(delay_ms, signal)


__all__ = [
    "DEFAULT_MAX_RETRY_DELAY_MS",
    "retry_provider_request",
]
