"""v4 InMemory Session 存储（对齐 TS `harness/session/memory.ts`）。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, cast

from pi_ai.utils.uuid import uuidv7

from .session import Session
from .state import SessionState
from .types import (
    BranchEntryQuery,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
    SessionStats,
)


def _now_ms() -> int:
    import time

    return time.time_ns() // 1_000_000


class InMemorySessionStorage:
    """纯内存 v4 会话存储：mutation 串行化 + 不可变读写。"""

    def __init__(self, metadata: SessionMetadata) -> None:
        self._metadata = deepcopy(metadata)
        self._state = SessionState()
        self._lock = asyncio.Lock()

    def fork(self, metadata: SessionMetadata, options: ForkOptions) -> "InMemorySessionStorage":
        storage = InMemorySessionStorage(metadata)
        for mutation in self._state.create_fork_mutations(options):
            storage._state.apply_mutation(mutation)
        return storage

    async def get_metadata(self) -> SessionMetadata:
        return deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        async with self._lock:
            self._state.validate_new_lane(lane)
            self._state.validate_target(at)
            self._state.apply_mutation(
                {
                    "kind": "lane",
                    "seq": self._state.next_sequence,
                    "lane": lane,
                    "leafId": at,
                }
            )

    async def move_lane(self, lane: str, to: str | None) -> None:
        async with self._lock:
            self._state.require_lane(lane)
            self._state.validate_target(to)
            self._state.apply_mutation(
                {
                    "kind": "lane",
                    "seq": self._state.next_sequence,
                    "lane": lane,
                    "leafId": to,
                }
            )

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        async with self._lock:
            parent_id = self._state.require_lane(lane)
            self._state.validate_unused_id(entry["id"])
            full = deepcopy(entry)
            full["parentId"] = parent_id
            full["seq"] = self._state.next_sequence
            full["timestamp"] = _now_ms()
            self._state.apply_mutation({"kind": "entry", "lane": lane, "entry": cast(Entry, full)})
            return cast(Entry, deepcopy(full))

    async def append_record(self, record: NewRecord) -> LaneRecord:
        async with self._lock:
            self._state.require_lane(record["lane"])
            self._state.validate_unused_id(record["id"])
            if record["type"] == "operation_started":
                open_operations = self._state.find_open_operations(record["lane"], {"limit": 1})
                if open_operations:
                    raise SessionError(
                        "storage",
                        f"Lane {record['lane']} already has an open operation {open_operations[0]['id']}",
                    )
            full = deepcopy(record)
            full["seq"] = self._state.next_sequence
            full["timestamp"] = _now_ms()
            self._state.apply_mutation({"kind": "record", "record": cast(LaneRecord, full)})
            return cast(LaneRecord, deepcopy(full))

    async def get_entry(self, entry_id: str) -> Entry | None:
        entry = self._state.get_entry(entry_id)
        return deepcopy(entry) if entry is not None else None

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return deepcopy(self._state.find_entries(query))

    async def find_entries_on_branch(self, query: BranchEntryQuery | None = None) -> list[Entry]:
        return deepcopy(self._state.find_entries_on_branch(dict(query or {})))

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return deepcopy(self._state.find_records(query))

    async def find_open_operations(
        self, lane: str, options: dict[str, int] | None = None
    ) -> list[OperationStartedRecord]:
        return deepcopy(self._state.find_open_operations(lane, options))

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return deepcopy(self._state.get_log(options))

    async def get_name(self) -> str | None:
        return self._state.get_name()

    async def set_name(self, name: str) -> None:
        async with self._lock:
            self._state.apply_mutation(
                {
                    "kind": "fact",
                    "seq": self._state.next_sequence,
                    "fact": "name",
                    "name": name,
                }
            )

    async def get_label(self, entry_id: str) -> str | None:
        return self._state.get_label(entry_id)

    async def set_label(self, entry_id: str, label: str | None) -> None:
        async with self._lock:
            self._state.validate_target(entry_id)
            self._state.apply_mutation(
                {
                    "kind": "fact",
                    "seq": self._state.next_sequence,
                    "fact": "label",
                    "targetId": entry_id,
                    "label": label,
                }
            )

    async def get_stats(self) -> SessionStats:
        return deepcopy(self._state.get_stats())


class InMemorySessionRepo:
    """纯内存 v4 会话仓库。"""

    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStorage] = {}

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        options = options or {}
        session_id = options.get("id") or uuidv7()
        if session_id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        metadata: SessionMetadata = {
            "id": session_id,
            "createdAt": _now_ms(),
        }
        if options.get("parentSessionId") is not None:
            metadata["parentSessionId"] = options["parentSessionId"]
        storage = InMemorySessionStorage(metadata)
        self._sessions[session_id] = storage
        return Session(storage)

    async def open(self, metadata: SessionMetadata) -> Session:
        return Session(self._require_storage(metadata["id"]))

    async def list(self, options: Any = None) -> list[SessionMetadata]:
        return [await storage.get_metadata() for storage in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata) -> None:
        self._sessions.pop(metadata["id"], None)

    async def fork(
        self,
        source: SessionMetadata,
        options: ForkOptions | None = None,
    ) -> Session:
        options = options or {}
        source_storage = self._require_storage(source["id"])
        session_id = options.get("id") or uuidv7()
        if session_id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        metadata: SessionMetadata = {
            "id": session_id,
            "createdAt": _now_ms(),
            "parentSessionId": cast(
                str,
                options.get("parentSessionId")
                if options.get("parentSessionId") is not None
                else source["id"],
            ),
        }
        storage = source_storage.fork(metadata, options)
        self._sessions[session_id] = storage
        return Session(storage)

    def _require_storage(self, session_id: str) -> InMemorySessionStorage:
        storage = self._sessions.get(session_id)
        if storage is None:
            raise SessionError("not_found", f"Session not found: {session_id}")
        return storage


__all__ = ["InMemorySessionStorage", "InMemorySessionRepo"]
