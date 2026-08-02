"""pi_ai.utils.retry 单元测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_ai._types import AssistantMessage
from pi_ai.utils.retry import (
    RetryCallbacks,
    RetryPolicy,
    compute_backoff_delay,
    is_retryable_error,
    retry_assistant_call,
)


def _error_msg(text: str) -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [],
        "api": "responses",
        "provider": "faux",
        "model": "faux",
        "stop_reason": "error",
        "error_message": text,
    }


def _ok_msg() -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [],
        "api": "responses",
        "provider": "faux",
        "model": "faux",
        "stop_reason": "stop",
    }


def _aborted_msg() -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [],
        "api": "responses",
        "provider": "faux",
        "model": "faux",
        "stop_reason": "aborted",
        "error_message": "Request was aborted",
    }


# ============================================================================
# is_retryable_error — 分类
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "429 Too Many Requests",
        "500 Internal Server Error",
        "503 Service Unavailable",
        "502 Bad Gateway",
        "504 Gateway Timeout",
        "The server is overloaded, please try again",
        "Rate limit exceeded: slow down",
        "rate_limit_reached",
        "too many requests",
        "network error: fetch failed",
        "ECONNREFUSED",
        "ENOTFOUND",
        "upstream connect error or disconnect/reset before headers",
        "socket hang up",
        "timed out",
        "request timeout",
        "websocket closed before receiving a response",
        "stream ended before message_stop",
        "http2 request did not get a response",
        "Retry delay of 2 seconds is recommended",
        "you can retry your request",
        "try your request again",
        "please retry your request",
        "ResourceExhausted: grpc",
        "Internal server error",
        "Connection refused",
        "eai_again",
    ],
)
def test_is_retryable_error_positive(text: str) -> None:
    assert is_retryable_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "insufficient_quota",
        "quota exceeded",
        "You exceeded your current quota",
        "billing issue on account",
        "GoUsageLimitError: monthly usage limit",
        "FreeUsageLimitError",
        "out of budget",
        "Monthly usage limit reached",
        "available balance is insufficient",
        # 不可重试优先：即便同时含 429 文本，quota 判定不可重试
        "429 quota exceeded",
    ],
)
def test_is_retryable_error_non_retryable(text: str) -> None:
    assert is_retryable_error(text) is False


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "Model not found: gpt-5",
        "Invalid API key provided",
        "content_filter",
    ],
)
def test_is_retryable_error_other(text: str | None) -> None:
    assert is_retryable_error(text) is False


def test_is_retryable_error_case_insensitive() -> None:
    assert is_retryable_error("RATE LIMIT EXCEEDED") is True
    assert is_retryable_error("QUOTA EXCEEDED") is False


# ============================================================================
# compute_backoff_delay — 退避数学
# ============================================================================


def test_backoff_exponential_no_jitter() -> None:
    assert compute_backoff_delay(1, 2000, jitter=False) == 2000
    assert compute_backoff_delay(2, 2000, jitter=False) == 4000
    assert compute_backoff_delay(3, 2000, jitter=False) == 8000


def test_backoff_capped_at_max_delay() -> None:
    assert compute_backoff_delay(10, 2000, max_delay_ms=60000, jitter=False) == 60000
    assert compute_backoff_delay(6, 2000, max_delay_ms=10000, jitter=False) == 10000


def test_backoff_jitter_within_bounds() -> None:
    for attempt in (1, 2, 3, 5, 10):
        delay = compute_backoff_delay(attempt, 2000, max_delay_ms=60000, jitter=True)
        cap = min(2000 * (2 ** (attempt - 1)), 60000)
        assert 0 <= delay <= cap


def test_backoff_attempt_zero_clamped() -> None:
    assert compute_backoff_delay(0, 2000, jitter=False) == 2000


# ============================================================================
# retry_assistant_call — 重试循环
# ============================================================================


async def test_retry_succeeds_on_second_attempt() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        if len(calls) == 1:
            return _error_msg("500 Internal Server Error")
        return _ok_msg()

    policy = RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "stop"
    assert len(calls) == 2


async def test_retry_exhausted_returns_last_error() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _error_msg("503 Service Unavailable")

    policy = RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "error"
    assert result["error_message"] == "503 Service Unavailable"
    # 初始调用 + 3 次重试
    assert len(calls) == 4


async def test_retry_zero_max_retries_no_retry() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _error_msg("500 Internal Server Error")

    policy = RetryPolicy(max_retries=0, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "error"
    assert len(calls) == 1


async def test_retry_disabled_policy() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _error_msg("500 Internal Server Error")

    policy = RetryPolicy(enabled=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "error"
    assert len(calls) == 1


async def test_retry_none_policy_no_retry() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _error_msg("500 Internal Server Error")

    result = await retry_assistant_call(produce, policy=None)

    assert result["stop_reason"] == "error"
    assert len(calls) == 1


async def test_retry_aborted_never_retries() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _aborted_msg()

    policy = RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "aborted"
    assert len(calls) == 1


async def test_retry_non_retryable_error_fast_fail() -> None:
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        return _error_msg("insufficient_quota")

    policy = RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy)

    assert result["stop_reason"] == "error"
    assert len(calls) == 1


async def test_retry_signal_aborts_during_sleep() -> None:
    calls: list[str] = []
    signal = asyncio.Event()

    async def produce() -> AssistantMessage:
        calls.append("call")
        if len(calls) == 1:
            return _error_msg("500 Internal Server Error")
        return _ok_msg()

    policy = RetryPolicy(max_retries=3, base_delay_ms=10000, jitter=False)

    async def _run() -> asyncio.Task[AssistantMessage]:
        task = asyncio.create_task(
            retry_assistant_call(produce, policy=policy, signal=signal)
        )
        # 等第一次调用完成进入 sleep 后中止
        while len(calls) < 1:
            await asyncio.sleep(0)
        signal.set()
        return task

    task = await _run()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 中止后不再发起第二次调用
    assert len(calls) == 1


async def test_retry_callbacks_invoked() -> None:
    scheduled: list[tuple[int, int, float, str]] = []
    attempt_starts: list[int] = []
    finished: list[tuple[bool, int, str | None]] = []
    calls: list[str] = []

    async def produce() -> AssistantMessage:
        calls.append("call")
        if len(calls) == 1:
            return _error_msg("500 Internal Server Error")
        return _ok_msg()

    callbacks = RetryCallbacks(
        on_retry_scheduled=lambda attempt, max_attempts, delay_ms, error_message: (
            scheduled.append((attempt, max_attempts, delay_ms, error_message))
        ),
        on_retry_attempt_start=lambda: attempt_starts.append(len(attempt_starts) + 1),
        on_retry_finished=lambda success, attempt, final_error: (
            finished.append((success, attempt, final_error))
        ),
    )

    policy = RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False)
    await retry_assistant_call(produce, policy=policy, callbacks=callbacks)

    assert len(scheduled) == 1
    assert scheduled[0][0] == 1  # attempt
    assert scheduled[0][1] == 3  # max_attempts
    assert scheduled[0][2] == 1  # delay_ms
    assert scheduled[0][3] == "500 Internal Server Error"
    assert len(attempt_starts) == 1
    # 第 1 次重试成功 → finished 报告 attempt=1（TS 语义：最终成功的重试序号）
    assert finished == [(True, 1, None)]


async def test_retry_callbacks_finished_on_exhaustion() -> None:
    finished: list[tuple[bool, int, str | None]] = []

    async def produce() -> AssistantMessage:
        return _error_msg("503 Service Unavailable")

    callbacks = RetryCallbacks(
        on_retry_finished=lambda success, attempt, final_error: (
            finished.append((success, attempt, final_error))
        ),
    )

    policy = RetryPolicy(max_retries=2, base_delay_ms=1, jitter=False)
    result = await retry_assistant_call(produce, policy=policy, callbacks=callbacks)

    assert result["stop_reason"] == "error"
    assert finished == [(False, 2, "503 Service Unavailable")]
