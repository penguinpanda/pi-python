"""JSONL v4 会话存储（对齐 TS `harness/session/jsonl/storage.ts`）。"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, cast

from .codec import (
    encode_header,
    encode_mutation,
    invalid_file,
    metadata_from_header,
    parse_header,
    parse_mutation,
)
from .fs import JsonlSessionRepoFileSystem, LocalFileSystem
from .jsonl_types import JsonlSessionMetadata, JsonlV4Header
from .state import SessionMutation, SessionState
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
    SessionError,
    SessionStats,
)


def _publish_file_atomically(
    destination: str,
    populate: Callable[[str], None],
    fs: JsonlSessionRepoFileSystem | None = None,
) -> None:
    """写入临时文件后原子 rename；失败时尽力清理临时文件并保留原文件。"""
    file_system = fs or LocalFileSystem()
    temp_path = f"{destination}.tmp"
    try:
        populate(temp_path)
        file_system.rename_file(temp_path, destination)
    except Exception:
        try:
            file_system.remove(temp_path, force=True)
        except OSError:
            pass
        raise


class JsonlSessionStorage:
    """JSONL v4 会话存储：写入串行化 + 原子发布 + torn-tail 修复。"""

    def __init__(
        self,
        path: str | Path,
        metadata: JsonlSessionMetadata,
        fs: JsonlSessionRepoFileSystem | None = None,
    ) -> None:
        self._path = Path(path)
        self._metadata = deepcopy(metadata)
        self._state = SessionState()
        self._lock = asyncio.Lock()
        self._fs = fs or LocalFileSystem()

    @staticmethod
    async def create(
        path: str,
        header: JsonlV4Header,
        fs: JsonlSessionRepoFileSystem | None = None,
    ) -> "JsonlSessionStorage":
        file_path = Path(path)
        file_system = fs or LocalFileSystem()
        file_system.write_file(file_path, encode_header(header))
        info = file_system.file_info(file_path)
        return JsonlSessionStorage(
            file_path,
            metadata_from_header(header, str(file_path), info.mtime_ms),
            file_system,
        )

    @staticmethod
    async def load(
        path: str,
        fs: JsonlSessionRepoFileSystem | None = None,
    ) -> "JsonlSessionStorage":
        file_path = Path(path)
        file_system = fs or LocalFileSystem()
        try:
            content = file_system.read_text_file(file_path)
        except FileNotFoundError as error:
            raise SessionError("not_found", f"Session not found: {path}", error) from error
        physical_lines = content.split("\n")
        if physical_lines and physical_lines[-1] == "":
            physical_lines.pop()
        if not physical_lines or not physical_lines[0]:
            raise invalid_file(str(file_path), 1, "is missing a header")
        header = parse_header(physical_lines[0], str(file_path))
        storage = JsonlSessionStorage(
            file_path,
            metadata_from_header(header, str(file_path), file_system.file_info(file_path).mtime_ms),
            file_system,
        )
        for index in range(1, len(physical_lines)):
            line = physical_lines[index]
            try:
                mutation = parse_mutation(line, str(file_path), index + 1)
            except SessionError as error:
                if index != len(physical_lines) - 1 or error.cause is None:
                    raise
                valid_prefix = "\n".join(physical_lines[:index]) + "\n"

                def _repair_torn_tail(temp: str, prefix: str = valid_prefix) -> None:
                    file_system.write_file(temp, prefix)

                _publish_file_atomically(
                    str(file_path),
                    _repair_torn_tail,
                    file_system,
                )
                return storage
            storage._apply_mutation(mutation, str(file_path), index + 1)
        if not content.endswith("\n"):
            file_system.append_file(file_path, "\n")
        return storage

    async def fork(
        self, path: str, header: JsonlV4Header, options: ForkOptions
    ) -> "JsonlSessionStorage":
        mutations = self._state.create_fork_mutations(options)
        # 发布前先在内存重放校验，失败时原文件保持不变。
        target_state = SessionState()
        for mutation in mutations:
            target_state.apply_mutation(mutation)

        def _populate(temp: str) -> None:
            self._fs.write_file(temp, encode_header(header))
            for mutation in mutations:
                self._fs.append_file(temp, encode_mutation(mutation))

        _publish_file_atomically(str(path), _populate, self._fs)
        return await JsonlSessionStorage.load(path, self._fs)

    async def drain(self) -> None:
        """等待当前串行化写入完成（对齐 TS `drain()`）。"""
        async with self._lock:
            return None

    async def get_metadata(self) -> JsonlSessionMetadata:
        return deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        async with self._lock:
            self._state.validate_new_lane(lane)
            self._state.validate_target(at)
            mutation: SessionMutation = {
                "kind": "lane",
                "seq": self._state.next_sequence,
                "lane": lane,
                "leafId": at,
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)

    async def move_lane(self, lane: str, to: str | None) -> None:
        async with self._lock:
            self._state.require_lane(lane)
            self._state.validate_target(to)
            mutation: SessionMutation = {
                "kind": "lane",
                "seq": self._state.next_sequence,
                "lane": lane,
                "leafId": to,
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        async with self._lock:
            parent_id = self._state.require_lane(lane)
            self._state.validate_unused_id(entry["id"])
            full = deepcopy(entry)
            full["parentId"] = parent_id
            full["seq"] = self._state.next_sequence
            full["timestamp"] = time.time_ns() // 1_000_000
            mutation: SessionMutation = {
                "kind": "entry",
                "lane": lane,
                "entry": cast(Entry, full),
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)
            return cast(Entry, deepcopy(full))

    async def append_record(self, record: NewRecord) -> LaneRecord:
        async with self._lock:
            self._state.require_lane(record["lane"])
            self._state.validate_unused_id(record["id"])
            open_operations = self._state.find_open_operations(record["lane"], {"limit": 1})
            if record["type"] == "operation_started" and open_operations:
                raise SessionError(
                    "storage",
                    f"Lane {record['lane']} already has an open operation {open_operations[0]['id']}",
                )
            full = deepcopy(record)
            full["seq"] = self._state.next_sequence
            full["timestamp"] = time.time_ns() // 1_000_000
            mutation: SessionMutation = {
                "kind": "record",
                "record": cast(LaneRecord, full),
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)
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
            mutation: SessionMutation = {
                "kind": "fact",
                "seq": self._state.next_sequence,
                "fact": "name",
                "name": name,
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)

    async def get_label(self, entry_id: str) -> str | None:
        return self._state.get_label(entry_id)

    async def set_label(self, entry_id: str, label: str | None) -> None:
        async with self._lock:
            self._state.validate_target(entry_id)
            mutation: SessionMutation = {
                "kind": "fact",
                "seq": self._state.next_sequence,
                "fact": "label",
                "targetId": entry_id,
                "label": label,
            }
            self._append_mutation(mutation)
            self._apply_mutation(mutation)

    async def get_stats(self) -> SessionStats:
        return deepcopy(self._state.get_stats())

    def _append_mutation(self, mutation: SessionMutation) -> None:
        self._fs.append_file(self._path, encode_mutation(mutation))

    def _apply_mutation(
        self,
        mutation: SessionMutation,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        def _invalid(message: str) -> None:
            raise invalid_file(
                path or str(self._path),
                line or self._state.next_sequence + 1,
                message,
            )

        self._state.apply_mutation(mutation, _invalid)


__all__ = ["JsonlSessionStorage", "_publish_file_atomically"]
