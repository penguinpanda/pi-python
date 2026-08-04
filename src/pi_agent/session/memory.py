"""内存 Session 存储（Phase 3.2）。

对齐 TS `harness/session/memory-storage.ts` 与 `memory-repo.ts`。
"""

from __future__ import annotations

from typing import Any, cast

from .repo import (
    create_session_id,
    create_timestamp,
    get_entries_to_fork,
    get_path_to_root_or_compaction,
    SessionRepo,
)
from .search import ScanningSessionSearch
from .session import _get_session_name, _get_session_stats
from .types import (
    LeafEntry,
    SessionCreateOptions,
    SessionEntryCursorOptions,
    SessionError,
    SessionForkOptions,
    SessionMetadata,
    SessionSnapshot,
    SessionTreeEntry,
)


def _update_label_cache(
    labels_by_id: dict[str, str],
    entry: SessionTreeEntry,
) -> None:
    if entry["type"] != "label":
        return
    label = (entry.get("label") or "").strip()
    target_id = cast(Any, entry)["targetId"]
    if label:
        labels_by_id[target_id] = label
    else:
        labels_by_id.pop(target_id, None)


def _build_labels_by_id(entries: list[SessionTreeEntry]) -> dict[str, str]:
    labels_by_id: dict[str, str] = {}
    for entry in entries:
        _update_label_cache(labels_by_id, entry)
    return labels_by_id


def _generate_entry_id(by_id: dict[str, SessionTreeEntry]) -> str:
    from pi_ai.utils.uuid import uuidv7

    for _ in range(100):
        entry_id = uuidv7()[-8:]
        if entry_id not in by_id:
            return entry_id
    return uuidv7()


def _leaf_id_after_entry(entry: SessionTreeEntry) -> str | None:
    return entry["targetId"] if entry["type"] == "leaf" else entry["id"]


class InMemorySessionStorage:
    """纯内存会话存储（测试与轻量场景）。"""

    def __init__(
        self,
        entries: list[SessionTreeEntry] | None = None,
        metadata: SessionMetadata | None = None,
    ) -> None:
        self._entries: list[SessionTreeEntry] = list(entries or [])
        self._by_id: dict[str, SessionTreeEntry] = {entry["id"]: entry for entry in self._entries}
        self._labels_by_id = _build_labels_by_id(self._entries)
        self._leaf_id: str | None = None
        for entry in self._entries:
            self._leaf_id = _leaf_id_after_entry(entry)
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        self._metadata = metadata or {
            "id": create_session_id(),
            "createdAt": create_timestamp(),
        }

    async def get_metadata(self) -> SessionMetadata:
        return dict(self._metadata)

    async def get_leaf_id(self) -> str | None:
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        return self._leaf_id

    async def set_leaf_id(self, leaf_id: str | None) -> LeafEntry:
        if leaf_id is not None and leaf_id not in self._by_id:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        entry: LeafEntry = {
            "type": "leaf",
            "id": _generate_entry_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": create_timestamp(),
            "targetId": leaf_id,
        }
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._leaf_id = leaf_id
        return entry

    async def create_entry_id(self) -> str:
        return _generate_entry_id(self._by_id)

    async def append_entry(self, entry: SessionTreeEntry) -> None:
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        _update_label_cache(self._labels_by_id, entry)
        self._leaf_id = _leaf_id_after_entry(entry)

    async def get_entry(self, entry_id: str) -> SessionTreeEntry | None:
        return self._by_id.get(entry_id)

    async def find_entries(self, entry_type: str) -> list[SessionTreeEntry]:
        return [entry for entry in self._entries if entry["type"] == entry_type]

    async def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

    async def get_session_name(self) -> str | None:
        return _get_session_name(self._entries)

    async def get_session_stats(self):
        return _get_session_stats(self._entries)

    async def get_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]:
        return get_path_to_root_or_compaction(self._entries, leaf_id)

    async def get_entries(
        self, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        start = (options or {}).get("afterEntrySeq", 0)
        limit = (options or {}).get("limit")
        end = None if limit is None else start + limit
        return self._entries[start:end]


class InMemorySessionStore:
    """内存会话存储仓库（对齐 TS InMemorySessionStore）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStorage] = {}

    async def create(self, options: SessionCreateOptions | None = None) -> SessionMetadata:
        options = options or {}
        metadata: SessionMetadata = {
            "id": options.get("id") or create_session_id(),
            "createdAt": create_timestamp(),
        }
        storage = InMemorySessionStorage(metadata=metadata)
        self._sessions[metadata["id"]] = storage
        return metadata

    async def open(self, metadata: SessionMetadata) -> InMemorySessionStorage:
        storage = self._sessions.get(metadata["id"])
        if storage is None:
            raise SessionError("not_found", f"Session not found: {metadata['id']}")
        return storage

    async def load(self, metadata: SessionMetadata) -> SessionSnapshot:
        storage = await self.open(metadata)
        return {
            "metadata": await storage.get_metadata(),
            "leafId": await storage.get_leaf_id(),
            "entries": await storage.get_entries(),
        }

    async def list(self) -> list[SessionMetadata]:
        return [await storage.get_metadata() for storage in self._sessions.values()]

    async def get_entries(
        self, metadata: SessionMetadata, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        return await (await self.open(metadata)).get_entries(options)

    async def create_entry_id(self, metadata: SessionMetadata) -> str:
        return await (await self.open(metadata)).create_entry_id()

    async def append_entry(self, metadata: SessionMetadata, entry: SessionTreeEntry) -> None:
        await (await self.open(metadata)).append_entry(entry)

    async def set_leaf_id(self, metadata: SessionMetadata, leaf_id: str | None) -> LeafEntry:
        return await (await self.open(metadata)).set_leaf_id(leaf_id)

    async def delete(self, metadata: SessionMetadata) -> None:
        self._sessions.pop(metadata["id"], None)

    async def fork(
        self, source: SessionMetadata, options: SessionForkOptions | None = None
    ) -> SessionMetadata:
        options = options or {}
        source_storage = await self.open(source)
        forked_entries = await get_entries_to_fork(source_storage, options)
        metadata: SessionMetadata = {
            "id": options.get("id") or create_session_id(),
            "createdAt": create_timestamp(),
        }
        storage = InMemorySessionStorage(entries=forked_entries, metadata=metadata)
        self._sessions[metadata["id"]] = storage
        return metadata


def create_in_memory_session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


def create_in_memory_session_repo() -> SessionRepo:
    store = create_in_memory_session_store()
    return SessionRepo(store=store, search=ScanningSessionSearch(store))
