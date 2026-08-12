"""v4 Session facade（对齐 TS `harness/session/session.ts`）。"""

from __future__ import annotations

import time
from typing import Any, Callable, cast

from pi_ai.utils.uuid import uuidv7

from ..._types import AgentMessage
from .context import (
    SessionContextBuildOptions,
    build_context_entries as build_v4_context_entries,
    build_session_context,
)
from .json_validation import assert_json_serializable
from .types import (
    BranchEntryQuery,
    Entry,
    EntryQuery,
    LaneRecord,
    LanePointer,
    LogItem,
    LogOptions,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionContext,
    SessionMetadata,
    SessionStats,
    SessionStorage,
    SessionTree,
)


def _assert_valid_limit(limit: Any) -> None:
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def _assert_valid_cursor(after_seq: Any) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


class LaneView:
    """绑定单个 lane 的会话视图（对齐 TS `Session.view(lane)`）。"""

    def __init__(self, session: "Session", lane: str) -> None:
        self._session = session
        self._lane = lane

    async def get_leaf_id(self) -> str | None:
        return await self._session.get_leaf_id_for_lane(self._lane)

    async def get_entry(self, entry_id: str) -> Entry | None:
        return await self._session.get_entry(entry_id)

    async def get_stats(self) -> SessionStats:
        return await self._session.get_stats()

    async def get_name(self) -> str | None:
        return await self._session.get_name()

    async def set_name(self, name: str) -> None:
        await self._session.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._session.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._session.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._session.find_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        return await self._session.find_entry(query)

    async def find_entries_on_branch(self, query: EntryQuery | None = None) -> list[Entry]:
        merged: dict[str, Any] = dict(query or {})
        _assert_valid_limit(merged.get("limit"))
        _assert_valid_cursor((merged.get("cursor") or {}).get("afterSeq"))
        if "start" not in merged:
            leaf = await self.get_leaf_id()
            if leaf is None:
                return []
            merged["start"] = leaf
        return await self._session.find_entries_on_branch(merged)

    async def find_entry_on_branch(self, query: EntryQuery | None = None) -> Entry | None:
        merged: dict[str, Any] = dict(query or {})
        _assert_valid_limit(merged.get("limit"))
        _assert_valid_cursor((merged.get("cursor") or {}).get("afterSeq"))
        if "start" not in merged:
            leaf = await self.get_leaf_id()
            if leaf is None:
                return None
            merged["start"] = leaf
        merged["limit"] = 1
        entries = await self._session.find_entries_on_branch(merged)
        return entries[0] if entries else None

    async def append_message(self, message: AgentMessage) -> str:
        return await self._session.append_message_to_lane(self._lane, message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._session.append_custom_entry_to_lane(self._lane, custom_type, data)


class Session:
    """v4 会话（实现 SessionTree；可创建/移动 lane）。"""

    def __init__(
        self,
        storage: SessionStorage,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._storage = storage
        self._id_generator = id_generator or uuidv7

    async def get_metadata(self) -> SessionMetadata:
        return await self._storage.get_metadata()

    def view(self, lane: str) -> SessionTree:
        return LaneView(self, lane)

    async def get_leaf_id(self) -> str | None:
        return await self.get_leaf_id_for_lane("main")

    async def get_leaf_id_for_lane(self, lane: str) -> str | None:
        lanes = await self._storage.get_lanes()
        for pointer in lanes:
            if pointer["lane"] == lane:
                return pointer["leafId"]
        raise SessionError("invalid_lane", f"Lane not found: {lane}")

    async def get_entry(self, entry_id: str) -> Entry | None:
        return await self._storage.get_entry(entry_id)

    async def get_stats(self) -> SessionStats:
        return await self._storage.get_stats()

    async def get_session_stats(self) -> SessionStats:
        """兼容别名。"""
        return await self.get_stats()

    async def get_name(self) -> str | None:
        return await self._storage.get_name()

    async def get_session_name(self) -> str | None:
        """兼容别名。"""
        return await self.get_name()

    async def set_name(self, name: str) -> None:
        await self._storage.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._storage.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._storage.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._storage.find_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        _assert_valid_limit((query or {}).get("limit"))
        entries = await self._storage.find_entries({**(query or {}), "limit": 1})
        return entries[0] if entries else None

    async def find_entries_on_branch(self, query: dict[str, Any] | None = None) -> list[Entry]:
        query = dict(query or {})
        _assert_valid_limit(query.get("limit"))
        _assert_valid_cursor((query.get("cursor") or {}).get("afterSeq"))
        if "start" not in query:
            start = await self.get_leaf_id()
            if start is None:
                return []
            query["start"] = start
        return await self._storage.find_entries_on_branch(cast(BranchEntryQuery, query))

    async def find_entry_on_branch(self, query: dict[str, Any] | None = None) -> Entry | None:
        query = dict(query or {})
        _assert_valid_limit(query.get("limit"))
        _assert_valid_cursor((query.get("cursor") or {}).get("afterSeq"))
        if "start" not in query:
            start = await self.get_leaf_id()
            if start is None:
                return None
            query["start"] = start
        query["limit"] = 1
        entries = await self._storage.find_entries_on_branch(cast(BranchEntryQuery, query))
        return entries[0] if entries else None

    async def get_lanes(self) -> list[LanePointer]:
        return await self._storage.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        await self._storage.create_lane(lane, at)

    async def move_lane(self, lane: str, to: str | None) -> None:
        await self._storage.move_lane(lane, to)

    async def append_message(self, message: AgentMessage) -> str:
        return await self.append_message_to_lane("main", message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self.append_custom_entry_to_lane("main", custom_type, data)

    async def get_branch(self, from_id: str | None = None) -> list[Entry]:
        """根到节点（默认 leaf）的路径，oldest-first（兼容入口）。"""
        start = from_id
        if start is None:
            start = await self.get_leaf_id()
        if start is None:
            return []
        return await self.find_entries_on_branch({"start": start, "order": "oldestFirst"})

    async def build_context_entries(
        self, options: SessionContextBuildOptions | None = None
    ) -> list[Entry]:
        return build_v4_context_entries(await self.get_branch(), options)

    async def build_context(
        self, options: SessionContextBuildOptions | None = None
    ) -> SessionContext:
        """把当前分支投影为 LLM 就绪上下文（兼容入口）。"""
        return build_session_context(await self.get_branch(), options)

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        entry = await self._commit_entry(
            {
                "type": "thinking_level_change",
                "id": self._id_generator(),
                "thinkingLevel": thinking_level,
            },
            "main",
        )
        return entry["id"]

    async def append_model_change(self, provider: str, model_id: str) -> str:
        entry = await self._commit_entry(
            {
                "type": "model_change",
                "id": self._id_generator(),
                "provider": provider,
                "modelId": model_id,
            },
            "main",
        )
        return entry["id"]

    async def append_active_tools_change(self, active_tool_names: list[str]) -> str:
        entry = await self._commit_entry(
            {
                "type": "active_tools_change",
                "id": self._id_generator(),
                "activeToolNames": list(active_tool_names),
            },
            "main",
        )
        return entry["id"]

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None = None,
        tokens_before: int = 0,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Any = None,
        retained_tail: list[AgentMessage] | None = None,
    ) -> str:
        """写入 compaction 条目；未提供 retainedTail 时由 firstKeptEntryId 推导。"""
        del from_hook  # v4 entry 无 fromHook 字段，仅保留签名兼容
        if retained_tail is None:
            retained_tail = []
            if first_kept_entry_id is not None:
                collecting = False
                for entry in await self.get_branch():
                    if entry["id"] == first_kept_entry_id:
                        collecting = True
                    if collecting and entry["type"] == "message":
                        retained_tail.append(cast(Any, entry)["message"])
        provisioned: dict[str, Any] = {
            "type": "compaction",
            "id": self._id_generator(),
            "summary": summary,
            "retainedTail": list(retained_tail),
            "tokensBefore": int(tokens_before or 0),
        }
        if details is not None:
            provisioned["details"] = details
        if usage is not None:
            provisioned["usage"] = usage
        entry = await self._commit_entry(provisioned, "main")
        return entry["id"]

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Any = None,
    ) -> str:
        """v4 无 custom_message 条目：映射为 message entry（role=custom）。"""
        message: dict[str, Any] = {
            "role": "custom",
            "customType": custom_type,
            "content": content,
            "display": display,
            "timestamp": time.time_ns() // 1_000_000,
        }
        if details is not None:
            message["details"] = details
        entry = await self._commit_entry(
            {
                "type": "message",
                "id": self._id_generator(),
                "message": cast(AgentMessage, message),
            },
            "main",
        )
        return entry["id"]

    async def append_label(self, target_id: str, label: str | None) -> str:
        """兼容入口：label 在 v4 中是 fact，返回 target_id。"""
        await self.set_label(target_id, label)
        return target_id

    async def append_session_name(self, name: str) -> str:
        """兼容入口：会话名在 v4 中是 fact。"""
        sanitized = " ".join(str(name).splitlines()).strip()
        await self.set_name(sanitized)
        return ""

    async def move_to(
        self, entry_id: str | None, summary: dict[str, Any] | None = None
    ) -> str | None:
        """移动 main lane 到指定条目，可选附带 branch_summary（兼容入口）。"""
        if entry_id is not None and await self.get_entry(entry_id) is None:
            raise SessionError("not_found", f"Entry not found: {entry_id}")
        await self.move_lane("main", entry_id)
        if not summary:
            return None
        provisioned: dict[str, Any] = {
            "type": "branch_summary",
            "id": self._id_generator(),
            "fromId": entry_id or "root",
            "summary": summary["summary"],
        }
        if summary.get("details") is not None:
            provisioned["details"] = summary["details"]
        if summary.get("usage") is not None:
            provisioned["usage"] = summary["usage"]
        entry = await self._commit_entry(provisioned, "main")
        return entry["id"]

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        return await self._commit_entry(entry, lane)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        return await self._commit_record(record)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        if query is not None and query.get("operationKind") is not None:
            if query.get("type") != "operation_started":
                raise SessionError(
                    "invalid_query", 'operationKind requires type "operation_started"'
                )
        return await self._storage.find_records(query)

    async def find_open_operations(
        self, lane: str, options: dict[str, int] | None = None
    ) -> list[OperationStartedRecord]:
        return await self._storage.find_open_operations(lane, options)

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return await self._storage.get_log(options)

    async def append_message_to_lane(self, lane: str, message: AgentMessage) -> str:
        entry = await self._commit_entry(
            {"type": "message", "id": self._id_generator(), "message": message}, lane
        )
        return entry["id"]

    async def append_custom_entry_to_lane(
        self, lane: str, custom_type: str, data: Any = None
    ) -> str:
        provisioned: ProvisionedEntry = {
            "type": "custom",
            "id": self._id_generator(),
            "customType": custom_type,
        }
        if data is not None:
            provisioned["data"] = data
        entry = await self._commit_entry(provisioned, lane)
        return entry["id"]

    async def _commit_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        assert_json_serializable(entry)
        return await self._storage.append_entry(entry, lane)

    async def _commit_record(self, record: NewRecord) -> LaneRecord:
        assert_json_serializable(record)
        return await self._storage.append_record(record)


__all__ = ["Session", "LaneView"]
