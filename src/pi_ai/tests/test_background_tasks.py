"""后台任务统一跟踪测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_ai.utils._background import pending_background_tasks, track_background_task


@pytest.mark.asyncio
async def test_track_background_task_completes_and_drops_reference() -> None:
    done = asyncio.Event()

    async def _work() -> None:
        done.set()

    task = track_background_task(_work())
    assert not task.done()
    assert task in pending_background_tasks()
    await asyncio.wait_for(done.wait(), timeout=5)
    await task
    assert task not in pending_background_tasks()


@pytest.mark.asyncio
async def test_track_background_task_survives_cancelled_stream() -> None:
    """任务异常不泄漏跟踪引用（done callback 移除）。"""
    done = asyncio.Event()

    async def _failing() -> None:
        try:
            raise RuntimeError("boom")
        finally:
            done.set()

    task = track_background_task(_failing())
    await asyncio.wait_for(done.wait(), timeout=5)
    with pytest.raises(RuntimeError):
        await task
    assert task not in pending_background_tasks()
