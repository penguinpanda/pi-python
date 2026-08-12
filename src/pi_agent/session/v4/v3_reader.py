"""只读 v3 JSONL 会话读取器，供 v3 → v4 惰性迁移使用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from typing_extensions import NotRequired

from pi_ai.types import ImageContent, TextContent, Usage

from ..._types import AgentMessage


SessionErrorCode = Literal[
    "not_found",
    "invalid_session",
    "invalid_entry",
    "invalid_fork_target",
    "storage",
    "unknown",
]


class SessionError(Exception):
    """v3 会话读取错误（迁移时转换为 v4 SessionError）。"""

    def __init__(
        self,
        code: SessionErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class _SessionTreeEntryBase(TypedDict):
    id: str
    parentId: str | None
    timestamp: str


class MessageEntry(_SessionTreeEntryBase):
    type: Literal["message"]
    message: AgentMessage


class ThinkingLevelChangeEntry(_SessionTreeEntryBase):
    type: Literal["thinking_level_change"]
    thinkingLevel: str


class ModelChangeEntry(_SessionTreeEntryBase):
    type: Literal["model_change"]
    provider: str
    modelId: str


class ActiveToolsChangeEntry(_SessionTreeEntryBase):
    type: Literal["active_tools_change"]
    activeToolNames: list[str]


class CompactionEntry(_SessionTreeEntryBase, total=False):
    type: Literal["compaction"]
    summary: str
    firstKeptEntryId: NotRequired[str | None]
    tokensBefore: NotRequired[int]
    retainedTail: NotRequired[list[AgentMessage]]
    details: NotRequired[Any]
    usage: NotRequired[Usage]
    fromHook: NotRequired[bool]


class BranchSummaryEntry(_SessionTreeEntryBase, total=False):
    type: Literal["branch_summary"]
    fromId: str
    summary: str
    details: NotRequired[Any]
    usage: NotRequired[Usage]
    fromHook: NotRequired[bool]


class CustomEntry(_SessionTreeEntryBase, total=False):
    type: Literal["custom"]
    customType: str
    data: NotRequired[Any]


class CustomMessageEntry(_SessionTreeEntryBase, total=False):
    type: Literal["custom_message"]
    customType: str
    content: str | list[TextContent | ImageContent]
    display: bool
    details: NotRequired[Any]


class LabelEntry(_SessionTreeEntryBase):
    type: Literal["label"]
    targetId: str
    label: str | None


class SessionInfoEntry(_SessionTreeEntryBase, total=False):
    type: Literal["session_info"]
    name: NotRequired[str]


class LeafEntry(_SessionTreeEntryBase):
    type: Literal["leaf"]
    targetId: str | None


SessionTreeEntry = (
    MessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | ActiveToolsChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
    | LeafEntry
)


class JsonlSessionMetadata(TypedDict):
    id: str
    createdAt: str
    cwd: str
    path: str
    parentSessionPath: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class SessionEntryCursorOptions(TypedDict, total=False):
    afterEntrySeq: NotRequired[int]
    limit: NotRequired[int]


def _leaf_id_after_entry(entry: SessionTreeEntry) -> str | None:
    return entry["targetId"] if entry["type"] == "leaf" else entry["id"]


def _invalid_session(
    file_path: str, message: str, cause: BaseException | None = None
) -> SessionError:
    return SessionError(
        "invalid_session",
        f"Invalid JSONL session file {file_path}: {message}",
        cause,
    )


def _invalid_entry(
    file_path: str, line_number: int, message: str, cause: BaseException | None = None
) -> SessionError:
    return SessionError(
        "invalid_entry",
        f"Invalid JSONL session file {file_path}: line {line_number} {message}",
        cause,
    )


def _parse_header_line(line: str, file_path: str) -> dict[str, Any]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise _invalid_session(
            file_path, "first line is not a valid session header", error
        ) from error
    if not isinstance(parsed, dict):
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("type") != "session":
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("version") != 3:
        raise _invalid_session(file_path, "unsupported session version")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_session(file_path, "session header is missing id")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_session(file_path, "session header is missing timestamp")
    if not isinstance(parsed.get("cwd"), str) or not parsed["cwd"]:
        raise _invalid_session(file_path, "session header is missing cwd")
    if parsed.get("parentSession") is not None and not isinstance(parsed.get("parentSession"), str):
        raise _invalid_session(file_path, "session header parentSession must be a string")
    return parsed


def _parse_entry_line(line: str, file_path: str, line_number: int) -> SessionTreeEntry:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise _invalid_entry(file_path, line_number, "is not valid JSON", error) from error
    if not isinstance(parsed, dict):
        raise _invalid_entry(file_path, line_number, "is not a valid session entry")
    if not isinstance(parsed.get("type"), str):
        raise _invalid_entry(file_path, line_number, "is missing entry type")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_entry(file_path, line_number, "is missing entry id")
    if parsed.get("parentId") is not None and not isinstance(parsed.get("parentId"), str):
        raise _invalid_entry(file_path, line_number, "has invalid parentId")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_entry(file_path, line_number, "is missing timestamp")
    return cast(SessionTreeEntry, parsed)


def _header_to_metadata(header: dict[str, Any], path: str) -> JsonlSessionMetadata:
    metadata: JsonlSessionMetadata = {
        "id": header["id"],
        "createdAt": header["timestamp"],
        "cwd": header["cwd"],
        "path": path,
    }
    if header.get("parentSession") is not None:
        metadata["parentSessionPath"] = header["parentSession"]
    if header.get("metadata") is not None:
        metadata["metadata"] = header["metadata"]
    return metadata


def _load_storage(
    file_path: Path,
) -> tuple[dict[str, Any], list[SessionTreeEntry], str | None]:
    content = file_path.read_text(encoding="utf-8")
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        raise _invalid_session(str(file_path), "missing session header")
    header = _parse_header_line(lines[0], str(file_path))
    entries: list[SessionTreeEntry] = []
    leaf_id: str | None = None
    for index, line in enumerate(lines[1:], start=2):
        entry = _parse_entry_line(line, str(file_path), index)
        entries.append(entry)
        leaf_id = _leaf_id_after_entry(entry)
    return header, entries, leaf_id


class JsonlSessionStorage:
    """只读 v3 JSONL 会话存储（迁移读取用）。"""

    def __init__(
        self,
        file_path: str,
        header: dict[str, Any],
        entries: list[SessionTreeEntry],
        leaf_id: str | None,
    ) -> None:
        self._file_path = Path(file_path)
        self._metadata = _header_to_metadata(header, str(self._file_path))
        self._entries = list(entries)
        self._by_id = {entry["id"]: entry for entry in self._entries}
        self._leaf_id = leaf_id

    @staticmethod
    async def open(file_path: str) -> "JsonlSessionStorage":
        header, entries, leaf_id = _load_storage(Path(file_path))
        return JsonlSessionStorage(file_path, header, entries, leaf_id)

    async def get_metadata(self) -> JsonlSessionMetadata:
        return cast(JsonlSessionMetadata, dict(self._metadata))

    async def get_leaf_id(self) -> str | None:
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        return self._leaf_id

    async def get_entries(
        self, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        start = (options or {}).get("afterEntrySeq", 0)
        limit = (options or {}).get("limit")
        end = None if limit is None else start + limit
        return self._entries[start:end]


__all__ = [
    "JsonlSessionStorage",
    "SessionError",
    "ActiveToolsChangeEntry",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "JsonlSessionMetadata",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "SessionEntryCursorOptions",
    "SessionInfoEntry",
    "SessionTreeEntry",
    "ThinkingLevelChangeEntry",
]
