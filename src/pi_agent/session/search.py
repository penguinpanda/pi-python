"""会话搜索（Phase 3.3）。

对齐 TS `harness/session/search-backend.ts` 与 `search-index.ts`。
"""

from __future__ import annotations

from typing import Protocol

from .repo import find_session_entry_matches
from .types import (
    SessionMetadata,
    SessionSearchHit,
    SessionSearchIndex,
    SessionSearchOptions,
    SessionSnapshot,
)


class SessionSearchSource(Protocol):
    """扫描式搜索的数据源（load + list）。"""

    async def load(self, metadata: SessionMetadata) -> SessionSnapshot: ...

    async def list(self) -> list[SessionMetadata]: ...


class ScanningSessionSearch:
    """直接扫描会话条目做搜索，无需维护索引。"""

    def __init__(self, source: SessionSearchSource) -> None:
        self._source = source

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        hits: list[SessionSearchHit] = []
        for metadata in await self._source.list():
            cwd = metadata.get("cwd")
            if options.get("cwd") is not None and cwd != options["cwd"]:
                continue
            state = await self._source.load(metadata)
            hits.extend(
                find_session_entry_matches(metadata, state["entries"], options.get("text", ""))
            )
        return hits


class SessionSearchIndexSource(Protocol):
    """重建索引的数据源。"""

    async def list(self) -> list[SessionMetadata]: ...

    async def load(self, metadata: SessionMetadata) -> SessionSnapshot: ...


async def rebuild_session_search_index(
    source: SessionSearchIndexSource,
    index: SessionSearchIndex,
) -> None:
    """全量重建搜索索引。"""
    for metadata in await source.list():
        entries = (await source.load(metadata))["entries"]
        await index.replace_session(metadata, entries)
