"""
pi_ai.utils.retry — 应用层重试原语。

对应 TypeScript `packages/ai/src/utils/retry.ts` 的移植：

    retryAssistantCall      → retry_assistant_call
    RetryPolicy             → RetryPolicy
    isRetryableAssistantError → is_retryable_error
    RetryCallbacks          → RetryCallbacks

设计要点：

    ① 错误表示模型：错误被编码为 AssistantMessage(stop_reason="error")
       携带 error_message 文本（无自定义异常类），分类靠正则匹配。

    ② 不可重试优先匹配：先查 quota/billing 类终态错误，再查可重试模式，
       避免 "rate limit"（可重试）与 "quota"（不可重试）混淆。

    ③ 中止语义：signal 为 asyncio.Event，置位即中止。退避等待期间
       signal 置位 → 抛 asyncio.CancelledError（与 agent loop 的
       _check_signal 语义一致，统一走既有 abort 路径）。

    ④ max_retries 语义：初始调用不计，max_retries 表示失败后最多重试
       次数（总调用数 = max_retries + 1）。max_retries=0 表示不重试。
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..types import AssistantMessage
from .diagnostics import append_assistant_message_diagnostic, create_assistant_message_diagnostic

# ============================================================================
# 错误分类
# ============================================================================

# 不可重试（终态）：配额 / 计费 / 使用量上限类错误，重试无意义。
# 与 TS NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN 对齐。
_NON_RETRYABLE_PATTERN = re.compile(
    r"(?:"
    r"GoUsageLimitError|"
    r"FreeUsageLimitError|"
    r"Monthly usage limit reached|"
    r"available balance|"
    r"insufficient_quota|"
    r"out of budget|"
    r"quota exceeded|"
    r"billing"
    r")",
    re.IGNORECASE,
)

# 可重试：过载 / 限流 / 5xx / 网络 / 流中断等瞬时故障。
# 与 TS RETRYABLE_PROVIDER_ERROR_PATTERN 对齐。
_RETRYABLE_PATTERN = re.compile(
    r"(?:"
    r"overloaded|"
    r"rate.?limit|"
    r"too many requests|"
    r"429|500|502|503|504|524|"
    r"service.?unavailable|"
    r"server.?error|"
    r"internal.?error|"
    r"provider.?returned.?error|"
    r"network.?error|"
    r"connection.?refused|"
    r"connection.?error|"
    r"connection.?lost|"
    r"other side closed|"
    r"fetch failed|"
    r"getaddrinfo|"
    r"ENOTFOUND|"
    r"ECONNREFUSED|"
    r"EAI_AGAIN|"
    r"upstream.?connect|"
    r"reset before headers|"
    r"socket hang up|"
    r"socket connection was closed|"
    r"timed? out|"
    r"timeout|"
    r"terminated|"
    r"websocket.?closed|"
    r"websocket.?error|"
    r"ended without|"
    r"stream ended before message_stop|"
    r"stream ended before a terminal response event|"
    r"http2 request did not get a response|"
    r"retry delay|"
    r"you can retry your request|"
    r"try your request again|"
    r"please retry your request|"
    r"ResourceExhausted"
    r")",
    re.IGNORECASE,
)


def is_retryable_error(error_message: str | None) -> bool:
    """判断错误消息文本是否可重试。

    不可重试优先匹配（quota/billing 等终态错误直接判定不可重试），
    再匹配可重试模式。None / 空串返回 False。
    """
    if not error_message:
        return False
    if _NON_RETRYABLE_PATTERN.search(error_message):
        return False
    return _RETRYABLE_PATTERN.search(error_message) is not None


# ============================================================================
# 退避策略
# ============================================================================


@dataclass(slots=True)
class RetryPolicy:
    """重试策略。

    对应 TS `RetryPolicy { enabled, maxRetries, baseDelayMs }`，
    额外增加 max_delay_ms（封顶）与 jitter（全抖动）两个开关。
    """

    enabled: bool = True
    # 失败后最多重试次数（初始调用不计；0 = 不重试）。
    max_retries: int = 3
    # 基础退避延迟（毫秒）。第 n 次重试延迟 = base * 2^(n-1)（封顶前）。
    base_delay_ms: float = 2000.0
    # 退避延迟上限（毫秒），默认 60s。
    max_delay_ms: float = 60000.0
    # 是否启用 full jitter（[0, cap] 均匀随机），默认 True。
    jitter: bool = True


def compute_backoff_delay(
    attempt: int,
    base_delay_ms: float = 2000.0,
    *,
    max_delay_ms: float = 60000.0,
    jitter: bool = True,
) -> float:
    """计算第 attempt 次重试的退避延迟（毫秒）。

    指数退避：cap = min(base * 2^(attempt-1), max_delay_ms)。
    jitter=True 时返回 [0, cap] 的 full jitter，否则返回 cap。
    """
    if attempt < 1:
        attempt = 1
    cap = min(base_delay_ms * (2 ** (attempt - 1)), max_delay_ms)
    if jitter:
        return random.uniform(0, cap)
    return cap


# ============================================================================
# 回调
# ============================================================================


@dataclass(slots=True)
class RetryCallbacks:
    """重试过程回调（对应 TS RetryCallbacks，全部可选）。"""

    # 退避 sleep 前：计划重试（attempt 从 1 起，max_attempts 为策略上限）。
    on_retry_scheduled: Callable[[int, int, float, str], None] | None = None
    # sleep 结束后、重试调用开始前。
    on_retry_attempt_start: Callable[[], None] | None = None
    # 重试循环结束（成功 / 放弃）各调用一次。
    on_retry_finished: Callable[[bool, int, str | None], None] | None = None


def _notify_scheduled(
    callbacks: RetryCallbacks | None,
    attempt: int,
    max_attempts: int,
    delay_ms: float,
    error_message: str,
) -> Awaitable[None] | None:
    """通知重试已计划；回调返回 coroutine 时交由调用方 await。"""
    if callbacks and callbacks.on_retry_scheduled:
        result = callbacks.on_retry_scheduled(attempt, max_attempts, delay_ms, error_message)
        if asyncio.iscoroutine(result):
            return result
    return None


def _notify_attempt_start(callbacks: RetryCallbacks | None) -> Awaitable[None] | None:
    """通知重试即将开始；回调返回 coroutine 时交由调用方 await。"""
    if callbacks and callbacks.on_retry_attempt_start:
        result = callbacks.on_retry_attempt_start()
        if asyncio.iscoroutine(result):
            return result
    return None


def _notify_finished(
    callbacks: RetryCallbacks | None,
    success: bool,
    attempt: int,
    final_error: str | None,
) -> Awaitable[None] | None:
    """通知重试循环结束；回调返回 coroutine 时交由调用方 await。"""
    if callbacks and callbacks.on_retry_finished:
        result = callbacks.on_retry_finished(success, attempt, final_error)
        if asyncio.iscoroutine(result):
            return result
    return None


async def _await_notify(coroutine: Awaitable[None] | None) -> None:
    """await 回调协程（如果有）。"""
    if coroutine is not None:
        await coroutine


# ============================================================================
# 单调用重试原语
# ============================================================================


async def _abortable_sleep(delay_s: float, signal: asyncio.Event | None) -> None:
    """可中止的 sleep。

    无 signal：直接 asyncio.sleep。
    有 signal：用 wait_for(signal.wait(), timeout) 等待；signal 置位时抛
    asyncio.CancelledError（与 agent loop 的 _check_signal 语义一致）。
    """
    if signal is None:
        await asyncio.sleep(delay_s)
        return
    try:
        await asyncio.wait_for(signal.wait(), timeout=delay_s)
    except asyncio.TimeoutError:
        return  # 睡满整个退避周期，未被中止
    # signal 已置位 → 中止
    raise asyncio.CancelledError()


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    *,
    policy: RetryPolicy | None = None,
    signal: asyncio.Event | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """用重试策略包裹一次完整的 AssistantMessage 生成调用。

    参数:
        produce: 单次调用，返回 AssistantMessage。
        policy: 重试策略；None 或 disabled 时等价于直接调 produce()。
        signal: 中止信号（asyncio.Event）；None 表示不可中止。
        callbacks: 重试过程回调（事件发射用）。

    行为（对齐 TS retryAssistantCall）:
        - stop_reason == "aborted" → 直接返回，永不重试中止。
        - stop_reason != "error"   → 成功返回。
        - 预算耗尽或错误不可重试   → 返回该错误（确定性错误快速失败）。
        - 否则按退避策略等待（可中止），然后重试。
    """
    if policy is None or not policy.enabled or policy.max_retries <= 0:
        return await produce()

    attempt = 0  # 已完成的尝试次数（初始调用为 0）
    while True:
        result = await produce()
        stop_reason = result.get("stop_reason")
        error_message = result.get("error_message") or "Unknown error"

        if stop_reason == "aborted":
            # 永不重试中止
            await _await_notify(_notify_finished(callbacks, False, attempt, error_message))
            return result
        if stop_reason != "error":
            # 成功（stop / length / tool_call / pending）
            await _await_notify(_notify_finished(callbacks, True, attempt, None))
            return result

        # 预算耗尽或不可重试 → 快速失败
        # 附加 retry_exhausted 诊断，记录放弃时的尝试次数（供调试定位）。
        if attempt >= policy.max_retries or not is_retryable_error(error_message):
            append_assistant_message_diagnostic(
                result,
                create_assistant_message_diagnostic(
                    "retry_exhausted",
                    error_message,
                    {"attempts": attempt, "max_retries": policy.max_retries},
                ),
            )
            await _await_notify(_notify_finished(callbacks, False, attempt, error_message))
            return result

        attempt += 1
        delay_ms = compute_backoff_delay(
            attempt,
            policy.base_delay_ms,
            max_delay_ms=policy.max_delay_ms,
            jitter=policy.jitter,
        )
        await _await_notify(
            _notify_scheduled(callbacks, attempt, policy.max_retries, delay_ms, error_message)
        )
        await _abortable_sleep(delay_ms / 1000.0, signal)
        await _await_notify(_notify_attempt_start(callbacks))
