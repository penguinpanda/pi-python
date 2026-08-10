"""v4 会话搜索（对齐 TS `createScanningSessionSearch`）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .jsonl_types import JsonlSessionListOptions
from .types import SessionSearchHit, SessionSearchOptions

if TYPE_CHECKING:
    from .repo import JsonlSessionRepo


class ScanningSessionSearch:
    """直接扫描 v4 会话条目做子串匹配（无需维护索引）。"""

    def __init__(self, repo: "JsonlSessionRepo") -> None:
        self._repo = repo

    async def search(self, options: SessionSearchOptions | None = None) -> list[SessionSearchHit]:
        options = options or {}
        normalized = (options.get("text") or "").strip().lower()
        if not normalized:
            return []
        list_options: JsonlSessionListOptions | None = (
            {"cwd": options["cwd"]} if options.get("cwd") is not None else None
        )
        hits: list[SessionSearchHit] = []
        for metadata in await self._repo.list(list_options):
            session = await self._repo.open(metadata)
            for entry in await session.find_entries({"order": "oldestFirst"}):
                payload = json.dumps(entry, ensure_ascii=False)
                if normalized not in payload.lower():
                    continue
                timestamp = (
                    datetime.fromtimestamp(entry["timestamp"] / 1000, tz=timezone.utc).isoformat()
                    if isinstance(entry["timestamp"], int)
                    else str(entry["timestamp"])
                )
                hits.append(
                    {
                        "metadata": metadata,
                        "entryId": entry["id"],
                        "timestamp": timestamp,
                        "snippet": payload,
                    }
                )
        return hits


__all__ = ["ScanningSessionSearch"]
