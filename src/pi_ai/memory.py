"""内存记忆存储（`MemoryStore` 协议的内置参考实现）。

`Context.memory` 目前只是扩展点；本模块提供可用的 dict 实现，
后续可替换为 Redis / 向量库等持久化后端。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


class InMemoryMemoryStore:
    """基于 dict 的 `MemoryStore` 参考实现（get/set/delete/search）。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        value = self._data.get(key)
        return deepcopy(value) if value is not None else None

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = deepcopy(value)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        """按 key 与序列化值做子串匹配（参考实现，非语义检索）。"""
        normalized = query.strip().lower()
        if not normalized:
            return []
        hits: list[Any] = []
        for key, value in self._data.items():
            if normalized in key.lower():
                hits.append(deepcopy(value))
            else:
                try:
                    serialized = json.dumps(value, ensure_ascii=False).lower()
                except (TypeError, ValueError):
                    serialized = ""
                if normalized in serialized:
                    hits.append(deepcopy(value))
            if len(hits) >= limit:
                break
        return hits


__all__ = ["InMemoryMemoryStore"]
