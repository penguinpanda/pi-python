"""设备代码 OAuth 轮询（对齐 TS auth/oauth/device-code.ts，RFC 8628）。"""

import asyncio
import math
import time
from typing import Any, Awaitable, Callable

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. "
    "This is often caused by clock drift in WSL or VM environments. "
    "Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
DEFAULT_POLL_INTERVAL_SECONDS = 5
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


async def abortable_sleep(ms: float, signal: asyncio.Event | None, cancel_message: str) -> None:
    """可中止的睡眠；signal 被 set 时抛 cancel_message。"""
    if signal is None:
        await asyncio.sleep(ms / 1000)
        return
    try:
        await asyncio.wait_for(signal.wait(), timeout=ms / 1000)
    except asyncio.TimeoutError:
        return
    if signal.is_set():
        raise RuntimeError(cancel_message)


async def poll_oauth_device_code_flow(
    poll: Callable[[], Awaitable[dict[str, Any]]],
    *,
    interval_seconds: float | None = None,
    expires_in_seconds: int | None = None,
    wait_before_first_poll: bool = False,
    signal: asyncio.Event | None = None,
) -> Any:
    """轮询设备代码授权结果；complete 返回 value，否则按状态推进。"""
    deadline = (
        time.time() * 1000 + expires_in_seconds * 1000
        if expires_in_seconds is not None
        else math.inf
    )
    interval_ms = max(
        MINIMUM_INTERVAL_MS,
        math.floor((interval_seconds or DEFAULT_POLL_INTERVAL_SECONDS) * 1000),
    )
    slow_down_responses = 0

    if wait_before_first_poll:
        remaining_ms = deadline - time.time() * 1000
        if remaining_ms > 0:
            await abortable_sleep(min(interval_ms, remaining_ms), signal, CANCEL_MESSAGE)

    while time.time() * 1000 < deadline:
        if signal is not None and signal.is_set():
            raise RuntimeError(CANCEL_MESSAGE)

        result = await poll()
        status = result.get("status")
        if status == "complete":
            return result.get("value")
        if status == "failed":
            raise RuntimeError(result.get("message", "Device flow failed"))
        if status == "slow_down":
            slow_down_responses += 1
            server_interval = result.get("interval_seconds")
            if (
                isinstance(server_interval, (int, float))
                and math.isfinite(server_interval)
                and server_interval > 0
            ):
                interval_ms = max(MINIMUM_INTERVAL_MS, math.floor(server_interval * 1000))
            else:
                interval_ms = max(
                    MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS
                )

        remaining_ms = deadline - time.time() * 1000
        if remaining_ms <= 0:
            break
        await abortable_sleep(min(interval_ms, remaining_ms), signal, CANCEL_MESSAGE)

    raise RuntimeError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)


__all__ = [
    "poll_oauth_device_code_flow",
    "abortable_sleep",
    "CANCEL_MESSAGE",
    "TIMEOUT_MESSAGE",
    "SLOW_DOWN_TIMEOUT_MESSAGE",
]
