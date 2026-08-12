"""Session Selector 业务模型（对齐 TS session-selector-search.ts + selector 行为）。"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pi_tui.engine.fuzzy import fuzzy_match

from ..._session_manager import SessionInfo
from ..._session_manager_v4 import open_session_manager


class SessionSortMode(str, Enum):
    """TS SortMode：threaded / recent / relevance（UI 显示 Fuzzy）。"""

    THREADED = "threaded"
    RECENT = "recent"
    RELEVANCE = "relevance"


class NameFilter(str, Enum):
    ALL = "all"
    NAMED = "named"


class SessionScope(str, Enum):
    CURRENT = "current"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class ParsedSearchQuery:
    mode: str
    tokens: tuple[tuple[str, str], ...] = ()
    regex: re.Pattern[str] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: bool
    score: float


@dataclass(slots=True)
class SessionDisplayNode:
    """展平后的树节点（depth / 连接线信息由 renderer 消费）。"""

    session: SessionInfo
    depth: int = 0
    is_last: bool = True
    ancestor_continues: tuple[bool, ...] = ()


@dataclass(frozen=True, slots=True)
class DeleteResult:
    ok: bool
    method: str
    error: str | None = None


@dataclass(slots=True)
class RenameResult:
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class SessionPickerModel:
    """Session Picker 的纯模型：排序 / 过滤 / scope / 选中项。"""

    current_sessions: list[SessionInfo]
    all_sessions: list[SessionInfo] | None = None
    current_cwd: str | None = None
    current_session_path: str | None = None
    sort_mode: SessionSortMode = SessionSortMode.THREADED
    name_filter: NameFilter = NameFilter.ALL
    scope: SessionScope = SessionScope.CURRENT
    show_path: bool = False
    query: str = ""
    rows: list[SessionDisplayNode] = field(default_factory=list, init=False)
    selected_index: int = 0

    def __post_init__(self) -> None:
        self.current_sessions = [_as_session_info(session) for session in self.current_sessions]
        self.all_sessions = (
            [_as_session_info(session) for session in self.all_sessions]
            if self.all_sessions is not None
            else list(self.current_sessions)
        )
        self._refresh()

    # ------------------------------------------------------------------
    # 状态切换
    # ------------------------------------------------------------------

    def toggle_scope(self) -> None:
        self.scope = (
            SessionScope.ALL if self.scope == SessionScope.CURRENT else SessionScope.CURRENT
        )
        self._refresh()

    def toggle_sort(self) -> None:
        self.sort_mode = {
            SessionSortMode.THREADED: SessionSortMode.RECENT,
            SessionSortMode.RECENT: SessionSortMode.RELEVANCE,
            SessionSortMode.RELEVANCE: SessionSortMode.THREADED,
        }[self.sort_mode]
        self._refresh()

    def toggle_name_filter(self) -> None:
        self.name_filter = (
            NameFilter.NAMED if self.name_filter == NameFilter.ALL else NameFilter.ALL
        )
        self._refresh()

    def toggle_path(self) -> None:
        self.show_path = not self.show_path

    def set_query(self, query: str) -> None:
        self.query = query
        self._refresh()

    def move_selection(self, delta: int) -> None:
        if not self.rows:
            return
        self.selected_index = min(len(self.rows) - 1, max(0, self.selected_index + delta))

    def page_selection(self, page_size: int) -> None:
        self.move_selection(page_size)

    @property
    def selected_node(self) -> SessionDisplayNode | None:
        if not self.rows:
            return None
        return self.rows[self.selected_index]

    @property
    def selected_path(self) -> str | None:
        node = self.selected_node
        return node.session.path if node is not None else None

    @property
    def is_selected_current(self) -> bool:
        path = self.selected_path
        return bool(
            path
            and self.current_session_path
            and _canonical(path) == _canonical(self.current_session_path)
        )

    def remove_path(self, path: str) -> None:
        """删除成功后从两个 scope 缓存中移除，避免刷新前重复出现。"""
        self.current_sessions = [
            session for session in self.current_sessions if session.path != path
        ]
        if self.all_sessions is not None:
            self.all_sessions = [session for session in self.all_sessions if session.path != path]
        self._refresh()

    def _visible_sessions(self) -> list[SessionInfo]:
        if self.scope == SessionScope.ALL and self.all_sessions is not None:
            return list(self.all_sessions)
        return list(self.current_sessions)

    def _refresh(self) -> None:
        sessions = self._visible_sessions()
        if self.sort_mode == SessionSortMode.THREADED and not self.query.strip():
            filtered = _apply_name_filter(sessions, self.name_filter)
            self.rows = flatten_session_tree(filtered)
        else:
            filtered = filter_and_sort_sessions(
                sessions,
                self.query,
                self.sort_mode,
                self.name_filter,
            )
            self.rows = [SessionDisplayNode(session=session) for session in filtered]
        self.selected_index = min(self.selected_index, max(0, len(self.rows) - 1))


def filter_and_sort_sessions(
    sessions: list[SessionInfo],
    query: str,
    sort_mode: SessionSortMode,
    name_filter: NameFilter = NameFilter.ALL,
) -> list[SessionInfo]:
    """对齐 TS filterAndSortSessions。"""
    name_filtered = _apply_name_filter(sessions, name_filter)
    trimmed = query.strip()
    if not trimmed:
        return name_filtered

    parsed = parse_search_query(query)
    if parsed.error:
        return []

    if sort_mode == SessionSortMode.RECENT:
        return [session for session in name_filtered if match_session(session, parsed).matches]

    scored: list[tuple[float, SessionInfo]] = []
    for session in name_filtered:
        result = match_session(session, parsed)
        if result.matches:
            scored.append((result.score, session))
    scored.sort(
        key=lambda pair: (pair[0], -pair[1].modified),
    )
    return [session for _score, session in scored]


def parse_search_query(query: str) -> ParsedSearchQuery:
    """解析普通 token / phrase / regex（对齐 TS parseSearchQuery）。"""
    trimmed = query.strip()
    if not trimmed:
        return ParsedSearchQuery(mode="tokens")

    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return ParsedSearchQuery(mode="regex", error="Empty regex")
        try:
            return ParsedSearchQuery(mode="regex", regex=re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            return ParsedSearchQuery(mode="regex", error=str(exc))

    tokens: list[tuple[str, str]] = []
    buf = ""
    in_quote = False
    had_unclosed_quote = False

    def flush(kind: str) -> None:
        nonlocal buf
        value = buf.strip()
        buf = ""
        if value:
            tokens.append((kind, value))

    for char in trimmed:
        if char == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue
        if not in_quote and char.isspace():
            flush("fuzzy")
            continue
        buf += char
    if in_quote:
        had_unclosed_quote = True

    if had_unclosed_quote:
        return ParsedSearchQuery(
            mode="tokens",
            tokens=tuple(("fuzzy", token) for token in trimmed.split() if token),
        )
    flush("phrase" if in_quote else "fuzzy")
    return ParsedSearchQuery(mode="tokens", tokens=tuple(tokens))


def match_session(session: SessionInfo, parsed: ParsedSearchQuery) -> MatchResult:
    """对齐 TS matchSession。"""
    text = session_search_text(session)

    if parsed.mode == "regex":
        if parsed.regex is None:
            return MatchResult(False, 0.0)
        match = parsed.regex.search(text)
        if match is None:
            return MatchResult(False, 0.0)
        return MatchResult(True, match.start() * 0.1)

    if not parsed.tokens:
        return MatchResult(True, 0.0)

    total = 0.0
    normalized_text: str | None = None
    for kind, token in parsed.tokens:
        if kind == "phrase":
            if normalized_text is None:
                normalized_text = _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(token)
            if not phrase:
                continue
            index = normalized_text.find(phrase)
            if index < 0:
                return MatchResult(False, 0.0)
            total += index * 0.1
            continue
        fuzzy_result = fuzzy_match(token, text)
        if not fuzzy_result.matches:
            return MatchResult(False, 0.0)
        total += fuzzy_result.score
    return MatchResult(True, total)


def has_session_name(session: SessionInfo) -> bool:
    return bool(session.name and session.name.strip())


def session_search_text(session: SessionInfo) -> str:
    if session.search_text:
        return session.search_text
    parts = [session.session_id, session.cwd]
    if session.name:
        parts.append(session.name)
    if session.first_message:
        parts.append(session.first_message)
    return " ".join(parts)


@dataclass(slots=True)
class _TreeSessionNode:
    session: SessionInfo
    children: list["_TreeSessionNode"] = field(default_factory=list)
    latest_activity: float = 0.0


def build_session_tree(sessions: list[SessionInfo]) -> list[_TreeSessionNode]:
    """按 parentSessionId 建树，子树按 latestActivity 倒序（对齐 TS buildSessionTree）。"""
    by_id: dict[str, _TreeSessionNode] = {}
    for session in sessions:
        by_id[session.session_id] = _TreeSessionNode(
            session=session,
            latest_activity=session.modified,
        )

    roots: list[_TreeSessionNode] = []
    for session in sessions:
        node = by_id[session.session_id]
        parent = by_id.get(session.parent_session_id or "")
        if parent is not None and parent is not node:
            parent.children.append(node)
        else:
            roots.append(node)

    def update_latest(node: _TreeSessionNode) -> float:
        latest = node.session.modified
        for child in node.children:
            latest = max(latest, update_latest(child))
        node.latest_activity = latest
        return latest

    for root in roots:
        update_latest(root)

    def sort_nodes(nodes: list[_TreeSessionNode]) -> None:
        nodes.sort(key=lambda node: node.latest_activity, reverse=True)
        for node in nodes:
            sort_nodes(node.children)

    sort_nodes(roots)
    return roots


def flatten_tree_nodes(roots: list[_TreeSessionNode]) -> list[SessionDisplayNode]:
    """把树展平为 display rows（对齐 TS flattenSessionTree）。"""
    rows: list[SessionDisplayNode] = []

    def walk(
        node: _TreeSessionNode,
        depth: int,
        ancestor_continues: tuple[bool, ...],
        is_last: bool,
    ) -> None:
        rows.append(
            SessionDisplayNode(
                session=node.session,
                depth=depth,
                is_last=is_last,
                ancestor_continues=ancestor_continues,
            )
        )
        for index, child in enumerate(node.children):
            child_is_last = index == len(node.children) - 1
            continues = (not is_last) if depth > 0 else False
            walk(
                child,
                depth + 1,
                (*ancestor_continues, continues),
                child_is_last,
            )

    for index, root in enumerate(roots):
        walk(root, 0, (), index == len(roots) - 1)
    return rows


def flatten_session_tree(sessions: list[SessionInfo]) -> list[SessionDisplayNode]:
    """公开的树形 flatten 入口（供测试与直接调用）。"""
    return flatten_tree_nodes(build_session_tree(sessions))


async def delete_session_file(session_path: str) -> DeleteResult:
    """优先 trash，失败回退 unlink（对齐 TS deleteSessionFile）。"""
    path = Path(session_path)
    trash = shutil.which("trash")
    if trash is not None:
        args = ["--", str(path)] if str(path).startswith("-") else [str(path)]
        try:
            process = await asyncio.create_subprocess_exec(
                trash,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()
            if process.returncode == 0 or not path.exists():
                return DeleteResult(ok=True, method="trash")
            error = stderr.decode(errors="replace").strip() or "trash exited with non-zero status"
            if process.returncode is not None and process.returncode != 0 and error:
                error = f"trash: {error}"
        except (OSError, subprocess.SubprocessError) as exc:
            error = f"trash: {exc}"
    else:
        error = "trash: not installed"
    try:
        path.unlink(missing_ok=True)
        Path(f"{path}.bak").unlink(missing_ok=True)
        return DeleteResult(ok=True, method="unlink")
    except OSError as exc:
        return DeleteResult(ok=False, method="unlink", error=f"{exc} ({error})")


async def rename_session_file(session_path: str, name: str) -> RenameResult:
    """重命名会话（对齐 TS appendSessionInfo）。"""
    next_name = name.strip()
    if not next_name:
        return RenameResult(ok=False, error="Session name cannot be empty")
    try:
        manager = await open_session_manager(session_path)
    except Exception as exc:
        return RenameResult(ok=False, error=str(exc))
    try:
        await manager.append_session_info(next_name)
    except Exception as exc:
        return RenameResult(ok=False, error=str(exc))
    finally:
        close = getattr(manager, "close", None)
        if close is not None:
            result = close()
            if result is not None and hasattr(result, "__await__"):
                await result
    return RenameResult(ok=True)


def _apply_name_filter(
    sessions: list[SessionInfo],
    name_filter: NameFilter,
) -> list[SessionInfo]:
    if name_filter == NameFilter.ALL:
        return list(sessions)
    return [session for session in sessions if has_session_name(session)]


def _normalize_whitespace_lower(text: str) -> str:
    return " ".join(text.lower().split())


def _canonical(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(Path(path))


def _as_session_info(session: Any) -> SessionInfo:
    if isinstance(session, SessionInfo):
        return session
    if isinstance(session, dict):
        return SessionInfo(
            path=str(session.get("path", "")),
            session_id=str(session.get("session_id", "")),
            cwd=str(session.get("cwd", "")),
            modified=float(session.get("modified", 0) or 0),
            name=session.get("name"),
            parent_session_id=session.get("parent_session_id") or session.get("parentSessionId"),
            first_message=str(session.get("first_message", "") or ""),
            message_count=int(session.get("message_count", 0) or 0),
            search_text=str(session.get("search_text", "") or ""),
        )
    return session


__all__ = [
    "DeleteResult",
    "MatchResult",
    "NameFilter",
    "ParsedSearchQuery",
    "RenameResult",
    "SessionDisplayNode",
    "SessionPickerModel",
    "SessionScope",
    "SessionSortMode",
    "build_session_tree",
    "delete_session_file",
    "filter_and_sort_sessions",
    "flatten_session_tree",
    "has_session_name",
    "match_session",
    "parse_search_query",
    "rename_session_file",
    "session_search_text",
]
