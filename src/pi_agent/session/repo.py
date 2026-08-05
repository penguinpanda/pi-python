"""Session 仓库与工具函数（Phase 3.2）。

对齐 TS `harness/session/repo-utils.ts` 与 `jsonl-repo.ts` 的仓库部分。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Generic, TypeVar, cast

from pi_ai.utils.uuid import uuidv7

from .session import (
    Session,
    get_path_to_root_or_compaction,
    _get_label,
    _get_session_name,
    _get_session_stats,
)
from .types import (
    LeafEntry,
    SessionCreateOptions,
    SessionEntryCursorOptions,
    SessionError,
    SessionForkOptions,
    SessionMetadata,
    SessionSearch,
    SessionSearchHit,
    SessionSearchOptions,
    SessionStats,
    SessionStorage,
    SessionStore,
    SessionTreeEntry,
)


def create_session_id() -> str:
    return uuidv7()


def create_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_session(storage: SessionStorage) -> Session:
    return Session(storage)


def _entries_by_id(entries: list[SessionTreeEntry]) -> dict[str, SessionTreeEntry]:
    return {entry["id"]: entry for entry in entries}


def find_session_entry_matches(
    metadata: SessionMetadata,
    entries: list[SessionTreeEntry],
    text: str,
) -> list[SessionSearchHit]:
    """在条目 JSON 中查找文本匹配（对齐 TS findSessionEntryMatches）。"""
    normalized = text.strip().lower()
    if not normalized:
        return []
    hits: list[SessionSearchHit] = []
    for entry in entries:
        payload = json.dumps(entry, ensure_ascii=False)
        if normalized not in payload.lower():
            continue
        hits.append(
            {
                "metadata": metadata,
                "entryId": entry["id"],
                "timestamp": entry["timestamp"],
                "snippet": payload,
            }
        )
    return hits


async def get_entries_to_fork(
    storage: SessionStorage,
    options: SessionForkOptions | None = None,
) -> list[SessionTreeEntry]:
    """计算 fork 要复制的条目路径。"""
    options = options or {}
    if not options.get("entryId"):
        return await storage.get_entries()
    target = await storage.get_entry(options["entryId"])
    if target is None:
        raise SessionError("invalid_fork_target", f"Entry {options['entryId']} not found")
    position = options.get("position", "before")
    if position == "at":
        effective_leaf_id: str | None = target["id"]
    else:
        if target["type"] != "message" or target["message"].get("role") != "user":
            raise SessionError(
                "invalid_fork_target",
                f"Entry {options['entryId']} is not a user message",
            )
        effective_leaf_id = target["parentId"]
    return await storage.get_path_to_root_or_compaction(effective_leaf_id)


TMetadata = TypeVar("TMetadata", bound=SessionMetadata)
TCreateOptions = TypeVar("TCreateOptions", bound=SessionCreateOptions)
TListOptions = TypeVar("TListOptions")


def to_store_session(
    store: SessionStore,
    metadata: SessionMetadata,
) -> Session:
    """把 store + metadata 包装成 Session（每次操作实时读 store）。"""

    def load():
        return store.load(metadata)

    class _StoreBackedStorage:
        async def get_metadata(self) -> SessionMetadata:
            return (await load())["metadata"]

        async def get_leaf_id(self) -> str | None:
            return (await load())["leafId"]

        async def set_leaf_id(self, leaf_id: str | None) -> LeafEntry:
            return await store.set_leaf_id(metadata, leaf_id)

        async def create_entry_id(self) -> str:
            return await store.create_entry_id(metadata)

        async def append_entry(self, entry: SessionTreeEntry) -> None:
            await store.append_entry(metadata, entry)

        async def get_entry(self, entry_id: str) -> SessionTreeEntry | None:
            return _entries_by_id((await load())["entries"]).get(entry_id)

        async def find_entries(self, entry_type: str) -> list[SessionTreeEntry]:
            return [entry for entry in (await load())["entries"] if entry["type"] == entry_type]

        async def get_label(self, entry_id: str) -> str | None:
            return _get_label((await load())["entries"], entry_id)

        async def get_session_name(self) -> str | None:
            return _get_session_name((await load())["entries"])

        async def get_session_stats(self) -> SessionStats:
            return _get_session_stats((await load())["entries"])

        async def get_path_to_root_or_compaction(
            self, leaf_id: str | None
        ) -> list[SessionTreeEntry]:
            return get_path_to_root_or_compaction((await load())["entries"], leaf_id)

        async def get_entries(
            self, options: SessionEntryCursorOptions | None = None
        ) -> list[SessionTreeEntry]:
            return await store.get_entries(metadata, options)

    return Session(_StoreBackedStorage())


class SessionRepo(Generic[TMetadata, TCreateOptions, TListOptions]):
    """组合 SessionStore + SessionSearch 的统一访问入口。"""

    def __init__(
        self,
        *,
        store: SessionStore,
        search: SessionSearch | None = None,
    ) -> None:
        self._store = store
        self._search_backend = search

    async def create(self, options: TCreateOptions | None = None) -> Session:
        metadata = await self._store.create(options or {})
        return to_store_session(self._store, metadata)

    async def open(self, metadata: TMetadata) -> Session:
        snapshot = await self._store.load(metadata)
        return to_store_session(self._store, snapshot["metadata"])

    async def delete(self, metadata: TMetadata) -> None:
        await self._store.delete(metadata)

    async def fork(self, source: TMetadata, options: SessionForkOptions) -> Session:
        metadata = await self._store.fork(source, options)
        return to_store_session(self._store, metadata)

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        if self._search_backend is None:
            return []
        return await self._search_backend.search(options)

    async def list(self, options: TListOptions | None = None) -> list[TMetadata]:
        return cast(list[TMetadata], await self._store.list())


def create_session_repo(
    *,
    store: SessionStore,
    search: SessionSearch | None = None,
) -> SessionRepo:
    return SessionRepo(store=store, search=search)
