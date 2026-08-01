"""
JSONL 会话持久化（最小核心版）

格式: 每行一个 JSON 对象，首行为 SessionHeader，后续为 SessionMessageEntry。
parentId 形成单链表。最小核心仅支持单链（无分支/树/压缩）。

用法:
    mgr = SessionManager.create(cwd="/path/to/project")
    mgr.append_message(msg)            # 追加消息
    messages = mgr.build_context()     # 重建消息列表
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pi_agent import AgentMessage

from ._types import (
    CURRENT_SESSION_VERSION,
    SessionHeader,
    SessionMessageEntry,
)

# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """JSONL 会话持久化管理器（最小核心版）。

    设计: append-only，每条消息一行 JSON。parentId 形成单链。
    """

    def __init__(
        self,
        *,
        cwd: str,
        session_id: str,
        session_path: Path | None = None,
        entries: list[SessionMessageEntry] | None = None,
    ):
        self._cwd = cwd
        self._session_id = session_id
        self._session_path = session_path  # None = 内存模式
        self._entries: list[SessionMessageEntry] = entries or []
        # 跟踪最新的 parentId（单链尾）
        self._leaf_parent_id: str | None = (
            self._entries[-1]["id"] if self._entries else None
        )

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        cwd: str,
        sessions_dir: str | Path | None = None,
        session_id: str | None = None,
    ) -> SessionManager:
        """创建新会话文件，写入 header。

        Args:
            cwd: 项目工作目录
            sessions_dir: 会话存储目录（None 则使用默认 ~/.pi/agent/sessions/）
            session_id: 会话 ID（None 则自动生成 UUID）
        """
        sid = session_id or uuid.uuid4().hex[:16]
        dir_path = Path(sessions_dir) if sessions_dir else _default_sessions_dir()
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / f"{sid}.jsonl"
        header: SessionHeader = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": sid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": cwd,
        }

        _write_jsonl_line_sync(filepath, header)
        return SessionManager(cwd=cwd, session_id=sid, session_path=filepath)

    @staticmethod
    def open(
        filepath: str | Path,
        cwd_override: str | None = None,
    ) -> SessionManager:
        """打开已有会话文件，读取所有条目。

        Args:
            filepath: JSONL 文件路径
            cwd_override: 覆盖 header 中的 cwd（可选）
        """
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Session file not found: {fp}")

        header, entries = _read_jsonl_sync(fp)
        cwd = cwd_override or header.get("cwd", str(Path.cwd()))
        return SessionManager(
            cwd=cwd,
            session_id=header["id"],
            session_path=fp,
            entries=entries,
        )

    @staticmethod
    def in_memory(cwd: str, session_id: str | None = None) -> SessionManager:
        """创建非持久化会话（内存模式）。"""
        sid = session_id or uuid.uuid4().hex[:16]
        return SessionManager(cwd=cwd, session_id=sid)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cwd(self) -> str:
        return self._cwd

    def is_persisted(self) -> bool:
        return self._session_path is not None

    async def append_message(self, message: AgentMessage) -> str:
        """追加一条消息到 JSONL，返回 entryId。

        自动设置 id/parentId/timestamp。
        """
        entry_id = uuid.uuid4().hex[:16]
        entry: SessionMessageEntry = {
            "type": "message",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        }

        self._entries.append(entry)
        self._leaf_parent_id = entry_id

        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)

        return entry_id

    def build_context(self) -> list[AgentMessage]:
        """沿单链重建消息列表（按时间顺序）。

        从第一个 entry 沿 parentId 链走到 leaf，收集所有消息。
        """
        if not self._entries:
            return []

        # 构建 id → entry 映射
        entry_map: dict[str, SessionMessageEntry] = {}
        for e in self._entries:
            entry_map[e["id"]] = e

        # 从 leaf 回溯到 root
        reversed_messages: list[AgentMessage] = []
        current_id = self._leaf_parent_id
        while current_id is not None:
            entry = entry_map.get(current_id)
            if entry is None:
                break
            reversed_messages.append(entry["message"])
            current_id = entry["parentId"]

        # 反转得到时间顺序
        reversed_messages.reverse()
        return reversed_messages

    def get_entries(self) -> list[SessionMessageEntry]:
        """返回所有条目（用于测试/检查）。"""
        return list(self._entries)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _default_sessions_dir() -> Path:
    """默认会话目录。"""
    from ._config import get_sessions_dir
    return get_sessions_dir()


def _write_jsonl_line_sync(filepath: Path, entry: dict[str, Any]) -> None:
    """同步写入单行 JSON（用于创建文件时写 header）。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _read_jsonl_sync(
    filepath: Path,
) -> tuple[SessionHeader, list[SessionMessageEntry]]:
    """同步读取 JSONL 文件，返回 (header, entries)。"""
    header: SessionHeader | None = None
    entries: list[SessionMessageEntry] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entry_type = obj.get("type")
            if entry_type == "session" and header is None:
                header = obj  # type: ignore[assignment]
            elif entry_type == "message":
                entries.append(obj)  # type: ignore[arg-type]

    if header is None:
        raise ValueError(f"Invalid session file: no header found in {filepath}")

    return header, entries


def _append_jsonl_line_sync_append(filepath: Path, entry: dict[str, Any]) -> None:
    """同步追加单行 JSON（用于追加消息）。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
