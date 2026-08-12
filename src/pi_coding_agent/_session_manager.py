"""v4 会话管理器的同步兼容门面（旧 v3 实现已移除）。

保留 `SessionManager` / `SessionInfo` / `SessionTreeNode` 名字，避免
调用方和旧测试大改；底层完全委托给 `_session_manager_v4`。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._types import SessionEntry


@dataclass(slots=True)
class SessionTreeNode:
    """会话树节点（get_tree 返回）。"""

    id: str
    parent_id: str | None
    children: list["SessionTreeNode"] = field(default_factory=list)
    entry: SessionEntry | None = None
    label: str | None = None
    label_timestamp: str | None = None


@dataclass(slots=True)
class SessionInfo:
    """list_sessions 返回的会话摘要。"""

    path: str
    session_id: str
    cwd: str
    modified: float
    name: str | None = None
    parent_session_id: str | None = None
    first_message: str = ""
    message_count: int = 0
    search_text: str = ""


def _run_sync(coro: Any) -> Any:
    """在当前或新事件循环中运行协程，兼容同步/异步调用方。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _schedule_task(coroutine: Any) -> None:
    """在运行中的事件循环里调度协程（无循环时跳过）。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    asyncio.create_task(coroutine)


class SessionManager:
    """v4 会话管理器同步门面，API 对齐旧 `SessionManager`。"""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    @staticmethod
    def create(
        cwd: str,
        sessions_dir: str | Path | None = None,
        session_id: str | None = None,
    ) -> "SessionManager":
        from ._session_manager_v4 import create_session_manager

        return SessionManager(_run_sync(create_session_manager(cwd, sessions_dir, session_id)))

    @staticmethod
    def in_memory(cwd: str, session_id: str | None = None) -> "SessionManager":
        from ._session_manager_v4 import in_memory_session_manager

        return SessionManager(_run_sync(in_memory_session_manager(cwd, session_id)))

    @staticmethod
    def open(filepath: str | Path, cwd_override: str | None = None) -> "SessionManager":
        from ._session_manager_v4 import open_session_manager

        expanded = str(Path(filepath).expanduser())
        return SessionManager(_run_sync(open_session_manager(expanded, cwd_override)))

    @staticmethod
    def list_sessions(directory: str | Path, cwd: str | None = None) -> list[SessionInfo]:
        from ._session_manager_v4 import list_sessions

        return _run_sync(list_sessions(directory, cwd))

    def fork(
        self,
        from_entry_id: str,
        *,
        position: str = "at",
        session_id: str | None = None,
        sessions_dir: str | Path | None = None,
    ) -> "SessionManager":
        return SessionManager(
            _run_sync(
                self._manager.fork(
                    from_entry_id,
                    position=position,
                    session_id=session_id,
                    sessions_dir=sessions_dir,
                )
            )
        )

    def edit_message(
        self,
        entry_id: str,
        new_text: str,
        *,
        mode: str = "merge",
    ) -> str:
        return _run_sync(self._manager.edit_message(entry_id, new_text, mode=mode))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


__all__ = ["SessionManager", "SessionInfo", "SessionTreeNode", "_schedule_task"]
