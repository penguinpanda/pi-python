"""文件变更序列化（对齐 TS harness/tools/file-mutation-queue.ts）。

同一文件（按 canonical path）的并发 write/edit 按注册顺序串行执行。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar, Union

from ..env import ExecutionEnv, get_or_throw

T = TypeVar("T")


class _MutationState:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Future] = {}
        self.registration: asyncio.Future = _done_future()


def _done_future() -> asyncio.Future:
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


_states: dict[int, _MutationState] = {}


def _get_state(env: ExecutionEnv) -> _MutationState:
    state = _states.get(id(env))
    if state is None:
        state = _MutationState()
        _states[id(env)] = state
    return state


async def _get_mutation_queue_key(env: ExecutionEnv, path: str) -> str:
    absolute = get_or_throw(await env.absolute_path(path))
    canonical = await env.canonical_path(absolute)
    if canonical[0]:
        return canonical[1]
    if canonical[1].code in ("not_found", "not_supported"):
        return absolute
    raise canonical[1]


async def with_file_mutation_queue(
    env: ExecutionEnv,
    path: str,
    operation: Callable[[], Any],
):
    """串行化同一文件路径上的变更操作。"""
    state = _get_state(env)
    key = await _get_mutation_queue_key(env, path)
    current_queue = state.queues.get(key)
    release_next: asyncio.Future = asyncio.get_running_loop().create_future()
    state.queues[key] = release_next
    if current_queue is not None and not current_queue.done():
        await current_queue
    try:
        return await operation()
    finally:
        if not release_next.done():
            release_next.set_result(None)
        if state.queues.get(key) is release_next:
            state.queues.pop(key, None)
