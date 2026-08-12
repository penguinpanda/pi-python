"""后台任务统一跟踪。

所有“create_task 后不保存句柄”的后台任务都应通过 `track_background_task`
创建：持有引用防止被 GC 静默取消，任务完成时自动移除。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def track_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """create_task 并持有引用；任务完成时自动从跟踪集中移除。"""
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def pending_background_tasks() -> list[asyncio.Task[Any]]:
    """当前未完成的后台任务快照（诊断 / 测试用）。"""
    return [task for task in _BACKGROUND_TASKS if not task.done()]


__all__ = ["track_background_task", "pending_background_tasks"]
