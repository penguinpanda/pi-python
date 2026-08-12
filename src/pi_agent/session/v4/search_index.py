"""v4 持久化会话搜索索引（Python 自有增强，非 TS 对齐项）。

TS 目前只有扫描式搜索；这里提供 JSON 持久化索引：
- 每个条目预存 `json.dumps(entry)` 文本，检索语义与扫描式完全一致
  （子串匹配），但无需逐个重读会话文件；
- `SessionSearchIndex` 接口（upsert / replace / delete）可接 SQLite 等后端。
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .types import (
    Entry,
    SessionMetadata,
    SessionSearchHit,
    SessionSearchOptions,
)


@dataclass(slots=True)
class SessionIndexSummary:
    """索引内单个会话的摘要数据。"""

    metadata: dict[str, Any]
    name: str | None = None
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


class PersistentSessionSearchIndex:
    """JSON 文件持久化的会话搜索索引。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # session_id -> {"metadata": {...}, "entries": {entry_id: {"timestamp": int, "text": str}}}
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            self._data = raw

    def _save(self) -> None:
        temp = f"{self._path}.tmp"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False)
        os.replace(temp, self._path)

    async def upsert_entry(self, metadata: SessionMetadata, entry: Entry) -> None:
        async with self._lock:
            bucket = self._data.setdefault(
                metadata["id"],
                {
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                    "entries": {},
                },
            )
            bucket["entries"][entry["id"]] = {
                "timestamp": entry.get("timestamp", 0),
                "text": json.dumps(entry, ensure_ascii=False),
            }
            self._save()

    async def replace_session(self, metadata: SessionMetadata, entries: list[Entry]) -> None:
        async with self._lock:
            bucket: dict[str, Any] = {
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
                "entries": {
                    entry["id"]: {
                        "timestamp": entry.get("timestamp", 0),
                        "text": json.dumps(entry, ensure_ascii=False),
                    }
                    for entry in entries
                },
            }
            name = metadata.get("name")
            if name is not None:
                bucket["name"] = name
            self._data[metadata["id"]] = bucket
            self._save()

    def is_populated(self) -> bool:
        return bool(self._data)

    def summaries(self) -> list[SessionIndexSummary]:
        result: list[SessionIndexSummary] = []
        for bucket in self._data.values():
            metadata = dict(bucket.get("metadata") or {})
            result.append(
                SessionIndexSummary(
                    metadata=metadata,
                    name=cast(str | None, bucket.get("name")),
                    entries=dict(bucket.get("entries") or {}),
                )
            )
        return result

    async def delete_session(self, metadata: SessionMetadata) -> None:
        async with self._lock:
            self._data.pop(metadata["id"], None)
            self._save()

    async def search(self, options: SessionSearchOptions | None = None) -> list[SessionSearchHit]:
        options = options or {}
        normalized = (options.get("text") or "").strip().lower()
        if not normalized:
            return []
        cwd = options.get("cwd")
        hits: list[SessionSearchHit] = []
        for _session_id, bucket in self._data.items():
            metadata = bucket.get("metadata") or {}
            if cwd is not None and metadata.get("cwd") != cwd:
                continue
            for entry_id, payload in (bucket.get("entries") or {}).items():
                if normalized not in payload.get("text", "").lower():
                    continue
                hits.append(
                    {
                        "metadata": cast(SessionMetadata, dict(metadata)),
                        "entryId": entry_id,
                        "timestamp": str(payload.get("timestamp", 0)),
                        "snippet": payload.get("text", ""),
                    }
                )
        return hits


async def rebuild_v4_search_index(repo: Any, index: PersistentSessionSearchIndex) -> None:
    """全量重建索引（v3 文件会按默认路径惰性转换为 v4）。"""
    for metadata in await repo.list():
        session = await repo.open(metadata)
        entries = await session.find_entries({"order": "oldestFirst"})
        enriched = dict(metadata)
        name = await session.get_name()
        if name is not None:
            enriched["name"] = name
        await index.replace_session(cast(SessionMetadata, enriched), entries)


__all__ = [
    "PersistentSessionSearchIndex",
    "SessionIndexSummary",
    "rebuild_v4_search_index",
]
