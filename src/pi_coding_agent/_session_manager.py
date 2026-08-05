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

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from filelock import FileLock
from pi_agent import AgentMessage
from pi_ai import now_ms

from ._types import (
    BranchSummaryEntry,
    CURRENT_SESSION_VERSION,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    LeafEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionHeader,
    SessionInfoEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)


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
        entries: list[SessionEntry] | None = None,
    ):
        self._cwd = cwd
        self._session_id = session_id
        self._session_path = session_path  # None = 内存模式
        self._entries: list[SessionEntry] = entries or []
        self._session_name: str | None = None
        # 跟踪最新的 parentId（单链尾）
        self._leaf_parent_id: str | None = self._entries[-1]["id"] if self._entries else None

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
        # 展开 ~（TUI slash 命令 / CLI 引号路径可能传入字面 ~）
        fp = Path(filepath).expanduser()
        if not fp.exists():
            raise FileNotFoundError(f"Session file not found: {fp}")

        header, entries = _read_jsonl_sync(fp)
        # 版本迁移：v1/v2 → v3（自动重写文件）。
        if header.get("version", 1) < CURRENT_SESSION_VERSION:
            if migrate_session_entries(header, entries):
                _write_jsonl_sync(fp, header, entries)
        cwd = cwd_override or header.get("cwd", str(Path.cwd()))
        manager = SessionManager(
            cwd=cwd,
            session_id=header["id"],
            session_path=fp,
            entries=entries,
        )
        manager._restore_leaf(entries)
        manager._restore_session_name(entries)
        return manager

    def _restore_leaf(self, entries: list[SessionEntry]) -> None:
        """从 leaf 条目恢复叶指针；无 leaf 条目时回退到链尾。"""
        for entry in reversed(entries):
            if entry.get("type") == "leaf":
                target = entry.get("targetId")
                self._leaf_parent_id = cast(str | None, target) if target else None
                return
        if entries:
            self._leaf_parent_id = cast(str | None, entries[-1].get("id"))

    def _restore_session_name(self, entries: list[SessionEntry]) -> None:
        for entry in reversed(entries):
            if entry.get("type") == "session_info" and entry.get("name"):
                self._session_name = cast(str, entry.get("name"))
                return

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
    def session_path(self) -> Path | None:
        return self._session_path

    @property
    def session_name(self) -> str | None:
        return self._session_name

    def set_session_name(self, name: str) -> None:
        """设置会话显示名（当前为进程内属性，不写入 JSONL）。"""
        self._session_name = name

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
        """沿当前 leaf 分支重建消息列表（按时间顺序）。

        遇到 compaction 条目时：产出 compactionSummary 消息并停止继续向下
        遍历（compaction 之前的旧历史已由摘要替代）；非消息条目
        （model_change / branch_summary / label 等）不参与上下文。
        """
        if not self._entries:
            return []

        # 构建 id → entry 映射
        entry_map: dict[str, SessionEntry] = {}
        for e in self._entries:
            entry_map[e["id"]] = e

        # 从 leaf 回溯到 root
        reversed_messages: list[AgentMessage] = []
        current_id = self._leaf_parent_id
        while current_id is not None:
            entry = entry_map.get(current_id)
            if entry is None:
                break
            if entry["type"] == "compaction":
                # 压缩条目：以 summary 消息替代旧历史，不再向下遍历。
                reversed_messages.append(
                    _compaction_summary_message(
                        entry["summary"],
                        entry.get("tokensBefore", 0),
                        entry["timestamp"],
                    )
                )
                break
            if entry["type"] == "custom_message":
                # 自定义消息：以 role=custom 进入上下文（display-only 字段保留给渲染器）。
                reversed_messages.append(
                    cast(
                        AgentMessage,
                        {
                            "role": "custom",
                            "content": entry.get("content"),
                            "customType": entry.get("customType", "custom"),
                            "display": entry.get("display", True),
                            "timestamp": entry.get("timestamp"),
                            "details": entry.get("details"),
                        },
                    )
                )
                current_id = entry["parentId"]
                continue
            message = entry.get("message")
            if message is not None:
                reversed_messages.append(cast(AgentMessage, message))
            current_id = entry["parentId"]

        # 反转得到时间顺序
        reversed_messages.reverse()
        return reversed_messages

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict | None = None,
    ) -> str:
        """追加一条压缩条目到 JSONL，返回 entryId。

        压缩条目挂在链尾，作为新历史的前缀；旧历史（firstKeptEntryId
        之前）仍保留在文件中但不再进入 build_context()。
        """
        entry_id = uuid.uuid4().hex[:16]
        entry: CompactionEntry = {
            "type": "compaction",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }

        self._entries.append(entry)
        self._leaf_parent_id = entry_id

        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)

        return entry_id

    async def append_model_change(self, provider: str, model_id: str) -> str:
        """追加一条模型切换条目到 JSONL，返回 entryId。"""
        entry_id = uuid.uuid4().hex[:16]
        entry: ModelChangeEntry = {
            "type": "model_change",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "modelId": model_id,
        }
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        """追加一条思考级别切换条目到 JSONL，返回 entryId。"""
        entry_id = uuid.uuid4().hex[:16]
        entry: ThinkingLevelChangeEntry = {
            "type": "thinking_level_change",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thinkingLevel": thinking_level,
        }
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    def get_entries(self) -> list[SessionEntry]:
        """返回所有条目（用于测试/检查）。"""
        return list(self._entries)

    def get_last_model_change(self) -> tuple[str, str] | None:
        """返回最后一条模型切换的 (provider, model_id)；无则 None。"""
        for entry in reversed(self._entries):
            if entry.get("type") == "model_change":
                model_id = entry.get("modelId") or entry.get("model_id")
                return str(entry.get("provider", "")), str(model_id or "")
        return None

    def get_leaf_id(self) -> str | None:
        """返回当前链尾条目 id（无条目时为 None）。"""
        return self._leaf_parent_id

    # ------------------------------------------------------------------
    # DAG 导航（Phase 6）
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        """按 id 查找条目。"""
        for entry in self._entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        """从根到指定条目（默认 leaf）的路径（含全部条目类型）。"""
        entry_map = {entry.get("id"): entry for entry in self._entries}
        path: list[SessionEntry] = []
        current_id = from_id if from_id is not None else self._leaf_parent_id
        current = entry_map.get(current_id) if current_id is not None else None
        while current is not None:
            path.append(current)
            parent_id = current.get("parentId")
            current = entry_map.get(parent_id) if parent_id is not None else None
        path.reverse()
        return path

    def get_tree(self) -> list[SessionTreeNode]:
        """构建会话树（孤儿条目作为根返回）。"""
        nodes: dict[str, SessionTreeNode] = {}
        for entry in self._entries:
            if entry.get("type") == "leaf":
                # leaf 是指针条目，不参与树结构。
                continue
            entry_id = cast(str, entry.get("id"))
            nodes[entry_id] = SessionTreeNode(
                id=entry_id,
                parent_id=entry.get("parentId"),
                entry=entry,
            )
        # 标签解析。
        for entry in self._entries:
            if entry.get("type") == "label":
                target = entry.get("targetId")
                node = nodes.get(cast(str, target))
                if node is not None:
                    node.label = cast(str | None, entry.get("label"))
                    node.label_timestamp = cast(str | None, entry.get("timestamp"))
        roots: list[SessionTreeNode] = []
        for entry in self._entries:
            if entry.get("type") == "leaf":
                continue
            node = nodes[cast(str, entry.get("id"))]
            parent_id = entry.get("parentId")
            if parent_id is None or parent_id == entry.get("id"):
                roots.append(node)
                continue
            parent = nodes.get(parent_id)
            if parent is not None:
                parent.children.append(node)
            else:
                roots.append(node)

        # 子节点按时间戳排序。
        def _sort(nodes_to_sort: list[SessionTreeNode]) -> None:
            for node in nodes_to_sort:
                node.children.sort(key=lambda child: _entry_ts(child.entry))
                _sort(node.children)

        _sort(roots)
        return roots

    def fork(
        self,
        from_entry_id: str,
        *,
        position: str = "at",
        session_id: str | None = None,
        sessions_dir: str | Path | None = None,
    ) -> "SessionManager":
        """从指定条目 fork 新分支：复制路径，创建新会话文件。"""
        path = self.get_branch(from_entry_id)
        if not path:
            raise ValueError(f"Entry not found: {from_entry_id}")
        kept = path if position == "at" else path[:-1]
        sid = session_id or uuid.uuid4().hex[:16]
        dir_path = Path(sessions_dir) if sessions_dir else _default_sessions_dir()
        dir_path.mkdir(parents=True, exist_ok=True)
        filepath = dir_path / f"{sid}.jsonl"
        header: SessionHeader = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": sid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": self._cwd,
        }
        _write_jsonl_sync(filepath, header, list(kept))
        manager = SessionManager(
            cwd=self._cwd,
            session_id=sid,
            session_path=filepath,
            entries=list(kept),
        )
        manager._leaf_parent_id = kept[-1].get("id") if kept else None
        return manager

    async def move_to(
        self, entry_id: str | None, summary: dict[str, Any] | None = None
    ) -> str | None:
        """移动 leaf 到指定条目；可选附带 branch_summary 条目。"""
        if entry_id is not None and self.get_entry(entry_id) is None:
            raise ValueError(f"Entry not found: {entry_id}")
        self._set_leaf(entry_id)
        if not summary:
            return None
        return await self.append_branch_summary(
            from_id=entry_id or "root",
            summary=summary.get("summary", ""),
            details=summary.get("details"),
            usage=summary.get("usage"),
            from_hook=bool(summary.get("fromHook", False)),
        )

    def _set_leaf(self, entry_id: str | None) -> None:
        """更新 leaf 并持久化 leaf 条目。"""
        self._leaf_parent_id = entry_id
        entry: LeafEntry = {
            "type": "leaf",
            "id": uuid.uuid4().hex[:16],
            "parentId": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "targetId": entry_id,
        }
        self._entries.append(entry)
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)

    # ------------------------------------------------------------------
    # 扩展条目（Phase 6）
    # ------------------------------------------------------------------

    async def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        *,
        details: Any = None,
        usage: Any = None,
        from_hook: bool = False,
    ) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry: BranchSummaryEntry = {
            "type": "branch_summary",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fromId": from_id,
            "summary": summary,
            "fromHook": from_hook,
        }
        if details is not None:
            entry["details"] = details
        if usage is not None:
            entry["usage"] = usage
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    async def append_label(self, target_id: str, label: str | None) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry: LabelEntry = {
            "type": "label",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "targetId": target_id,
            "label": label,
        }
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    def set_label(self, target_id: str, label: str | None) -> None:
        """同步设置标签（异步持久化由调用方决定）。"""
        _schedule_task(self.append_label(target_id, label))

    async def append_session_info(self, name: str | None) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry: SessionInfoEntry = {
            "type": "session_info",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if name is not None:
            entry["name"] = name
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        self._session_name = name
        return entry_id

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry: CustomEntry = {
            "type": "custom",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customType": custom_type,
        }
        if data is not None:
            entry["data"] = data
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        *,
        display: bool = True,
        details: Any = None,
    ) -> str:
        entry_id = uuid.uuid4().hex[:16]
        entry: CustomMessageEntry = {
            "type": "custom_message",
            "id": entry_id,
            "parentId": self._leaf_parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customType": custom_type,
            "content": content,
            "display": display,
        }
        if details is not None:
            entry["details"] = details
        self._entries.append(entry)
        self._leaf_parent_id = entry_id
        if self._session_path:
            _append_jsonl_line_sync_append(self._session_path, entry)
        return entry_id

    # ------------------------------------------------------------------
    # 会话发现 + 文件锁
    # ------------------------------------------------------------------

    @staticmethod
    def list_sessions(directory: str | Path) -> list[SessionInfo]:
        """扫描会话目录，按修改时间倒序返回。"""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []
        results: list[SessionInfo] = []
        for path in dir_path.glob("*.jsonl"):
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            session_id = path.stem
            cwd = ""
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    first = handle.readline().strip()
                header = json.loads(first)
                if isinstance(header, dict):
                    if header.get("id"):
                        session_id = header["id"]
                    cwd = header.get("cwd", "")
            except (OSError, json.JSONDecodeError):
                pass
            results.append(
                SessionInfo(
                    path=str(path),
                    session_id=session_id,
                    cwd=cwd,
                    modified=modified,
                )
            )
        results.sort(key=lambda info: info.modified, reverse=True)
        return results

    async def with_lock(self, fn):
        """文件级锁保护（filelock）。"""
        import inspect

        async def _run() -> Any:
            result = fn(self)
            if inspect.isawaitable(result):
                return await result
            return result

        if self._session_path is None:
            return await _run()
        lock = FileLock(str(self._session_path) + ".lock", timeout=30)
        lock.acquire()
        try:
            return await _run()
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _default_sessions_dir() -> Path:
    """默认会话目录。"""
    from ._config import get_sessions_dir

    return get_sessions_dir()


def _schedule_task(coroutine) -> None:
    """在运行中的事件循环里调度协程（无循环时跳过）。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    asyncio.create_task(coroutine)


def _entry_ts(entry: SessionEntry | None) -> float:
    if entry is None:
        return 0.0
    try:
        return datetime.fromisoformat(entry["timestamp"]).timestamp()
    except (ValueError, TypeError, KeyError):
        return 0.0


def _write_jsonl_sync(filepath: Path, header: Any, entries: list[Any]) -> None:
    """重写会话文件（header + entries）。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _write_jsonl_line_sync(filepath: Path, entry: Any) -> None:
    """同步写入单行 JSON（用于创建文件时写 header）。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _generate_entry_id(ids: set[str]) -> str:
    while True:
        candidate = uuid.uuid4().hex[:16]
        if candidate not in ids:
            ids.add(candidate)
            return candidate


def _migrate_v1_to_v2(header: dict[str, Any], entries: list[SessionEntry]) -> None:
    """v1 → v2：为条目补 id/parentId 树形结构（对齐 TS migrateV1ToV2）。"""
    header["version"] = 2
    ids: set[str] = set()
    prev_id: str | None = None
    for entry in entries:
        entry["id"] = _generate_entry_id(ids)
        entry["parentId"] = prev_id
        prev_id = entry.get("id")
        if entry.get("type") == "compaction":
            compaction = cast(CompactionEntry, entry)
            first_kept_index = entry.get("firstKeptEntryIndex")
            if isinstance(first_kept_index, int):
                # TS 的索引包含文件头（下标 0）；本列表不含头，减 1 对齐。
                target_index = first_kept_index - 1
                if 0 <= target_index < len(entries):
                    target = entries[target_index]
                    if target.get("type") != "session":
                        compaction["firstKeptEntryId"] = cast(str, target.get("id"))
                cast(dict[str, Any], compaction).pop("firstKeptEntryIndex", None)


def _migrate_v2_to_v3(header: dict[str, Any], entries: list[SessionEntry]) -> None:
    """v2 → v3：hookMessage 角色更名为 custom（对齐 TS migrateV2ToV3）。"""
    header["version"] = 3
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if isinstance(message, dict) and message.get("role") == "hookMessage":
            message["role"] = "custom"


def migrate_session_entries(header: Any, entries: list[SessionEntry]) -> bool:
    """按版本迁移条目到 CURRENT_SESSION_VERSION；返回是否发生迁移。"""
    version = header.get("version", 1)
    if version >= CURRENT_SESSION_VERSION:
        return False
    if version < 2:
        _migrate_v1_to_v2(header, entries)
    if header.get("version", 1) < 3:
        _migrate_v2_to_v3(header, entries)
    return True


def _read_jsonl_sync(
    filepath: Path,
) -> tuple[SessionHeader, list[SessionEntry]]:
    """同步读取 JSONL 文件，返回 (header, entries)。"""
    header: SessionHeader | None = None
    entries: list[SessionEntry] = []

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
            elif entry_type == "compaction":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "model_change":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "thinking_level_change":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "branch_summary":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "label":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "session_info":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "custom":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "custom_message":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "leaf":
                entries.append(obj)  # type: ignore[arg-type]
            elif entry_type == "active_tools_change":
                entries.append(obj)  # type: ignore[arg-type]

    if header is None:
        raise ValueError(f"Invalid session file: no header found in {filepath}")

    return header, entries


def _compaction_summary_message(
    summary: str,
    tokens_before: int,
    timestamp_iso: str,
) -> AgentMessage:
    """把压缩条目转换为上下文中的 compactionSummary 消息（对齐 TS）。"""
    try:
        ts = int(datetime.fromisoformat(timestamp_iso).timestamp() * 1000)
    except (ValueError, TypeError):
        ts = now_ms()
    return cast(
        AgentMessage,
        {
            "role": "compactionSummary",
            "summary": summary,
            "tokens_before": tokens_before,
            "timestamp": ts,
        },
    )


def _append_jsonl_line_sync_append(filepath: Path, entry: Any) -> None:
    """同步追加单行 JSON（用于追加消息）。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
