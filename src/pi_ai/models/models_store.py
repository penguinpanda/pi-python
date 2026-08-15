"""模型目录持久化（对齐 TS models-store.ts）。

- ModelsStoreEntry：models + lastModified + checkedAt + etag；
- ModelsStore：按 provider ID 的持久化接口；
- ProviderModelsStore：只暴露当前 provider 的读写（按 provider 隔离）；
- InMemoryModelsStore / FileModelsStore：内存与 auth.json 式文件实现。
"""

import copy
import json

from dataclasses import dataclass
from filelock import FileLock
from pathlib import Path
from typing import Any, Protocol

from ..types import Model
from ..utils.atomic_write import atomic_write_json


@dataclass(slots=True)
class ModelsStoreEntry:
    """单个 provider 的模型目录缓存条目。"""

    models: list[Model]
    # 远端目录 Last-Modified 对应的 Unix 毫秒时间戳。
    last_modified: int | None = None
    # 最近一次完成远端检查的 Unix 毫秒时间戳。
    checked_at: int | None = None
    # 远端目录 ETag（原样保存，含引号），回显为 If-None-Match。
    etag: str | None = None


class ModelsStore(Protocol):
    """按 provider ID 持久化的模型目录存储。"""

    async def read(self, provider_id: str) -> ModelsStoreEntry | None: ...

    async def write(self, provider_id: str, entry: ModelsStoreEntry) -> None: ...

    async def delete(self, provider_id: str) -> None: ...


class ProviderModelsStore(Protocol):
    """限定到单个 provider 的存储视图（provider 无法访问其它 provider）。"""

    async def read(self) -> ModelsStoreEntry | None: ...

    async def write(self, entry: ModelsStoreEntry) -> None: ...

    async def delete(self) -> None: ...


class _ProviderScopedStore:
    def __init__(self, store: ModelsStore, provider_id: str) -> None:
        self._store = store
        self._provider_id = provider_id

    async def read(self) -> ModelsStoreEntry | None:
        return await self._store.read(self._provider_id)

    async def write(self, entry: ModelsStoreEntry) -> None:
        await self._store.write(self._provider_id, entry)

    async def delete(self) -> None:
        await self._store.delete(self._provider_id)


def provider_models_store(store: ModelsStore, provider_id: str) -> ProviderModelsStore:
    """创建 provider 作用域的存储视图。"""
    return _ProviderScopedStore(store, provider_id)


class InMemoryModelsStore:
    """内存实现（生命周期限当前进程）。"""

    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(self, provider_id: str) -> ModelsStoreEntry | None:
        entry = self._entries.get(provider_id)
        return copy.deepcopy(entry) if entry is not None else None

    async def write(self, provider_id: str, entry: ModelsStoreEntry) -> None:
        self._entries[provider_id] = copy.deepcopy(entry)

    async def delete(self, provider_id: str) -> None:
        self._entries.pop(provider_id, None)


# ---------------------------------------------------------------
# Model 序列化（JSON 持久化共用）
# ---------------------------------------------------------------


def model_to_dict(model: Model) -> dict[str, Any]:
    """Model → 可 JSON 序列化的字典。"""
    cost = model.cost
    if cost is None:
        cost_dict = {
            "input": 0.0,
            "output": 0.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
            "tiers": [],
        }
    else:
        cost_dict = {
            "input": cost.input,
            "output": cost.output,
            "cache_read": cost.cache_read,
            "cache_write": cost.cache_write,
            "tiers": [
                {
                    "input": tier.input,
                    "output": tier.output,
                    "cache_read": tier.cache_read,
                    "cache_write": tier.cache_write,
                    "input_tokens_above": tier.input_tokens_above,
                }
                for tier in cost.tiers
            ],
        }
    return {
        "id": model.id,
        "provider": model.provider,
        "api": model.api,
        "name": model.name,
        "input": list(model.input),
        "output": list(model.output),
        "cost": cost_dict,
        "max_tokens": model.max_tokens,
        "base_url": model.base_url,
        "context_window": model.context_window,
        "headers": dict(model.headers) if model.headers else None,
        "compat": dict(model.compat) if model.compat else None,
        "thinking_level_map": (
            dict(model.thinking_level_map) if model.thinking_level_map else None
        ),
        "reasoning": model.reasoning,
        "deprecated": model.deprecated,
    }


def model_from_dict(data: dict[str, Any]) -> Model:
    """字典 → Model（model_to_dict 的逆操作）。"""
    from ..types import ModelCost, ModelCostTier

    cost_data = data.get("cost") or {}
    tiers = [
        ModelCostTier(
            input=tier.get("input", 0.0),
            output=tier.get("output", 0.0),
            cache_read=tier.get("cache_read", 0.0),
            cache_write=tier.get("cache_write", 0.0),
            input_tokens_above=tier.get("input_tokens_above", 0),
        )
        for tier in cost_data.get("tiers") or []
    ]
    return Model(
        id=data["id"],
        provider=data["provider"],
        api=data["api"],
        name=data.get("name", ""),
        input=list(data.get("input") or []),
        output=list(data.get("output") or []),
        cost=ModelCost(
            input=cost_data.get("input", 0.0),
            output=cost_data.get("output", 0.0),
            cache_read=cost_data.get("cache_read", 0.0),
            cache_write=cost_data.get("cache_write", 0.0),
            tiers=tiers,
        ),
        max_tokens=data.get("max_tokens", 4096),
        base_url=data.get("base_url", ""),
        context_window=data.get("context_window", 0),
        headers=data.get("headers"),
        compat=data.get("compat"),
        thinking_level_map=data.get("thinking_level_map"),
        reasoning=bool(data.get("reasoning", False)),
        deprecated=bool(data.get("deprecated", False)),
    )


class FileModelsStore:
    """JSON 文件持久化（原子写；适合 auth.json 风格的 ~/.pi 目录）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file_lock = FileLock(str(self._path) + ".lock", timeout=30)

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self._path, data)

    async def read(self, provider_id: str) -> ModelsStoreEntry | None:
        with self._file_lock:
            raw = self._load().get(provider_id)
        if raw is None:
            return None
        return ModelsStoreEntry(
            models=[model_from_dict(m) for m in raw.get("models", [])],
            last_modified=raw.get("last_modified"),
            checked_at=raw.get("checked_at"),
            etag=raw.get("etag"),
        )

    async def write(self, provider_id: str, entry: ModelsStoreEntry) -> None:
        with self._file_lock:
            data = self._load()
            data[provider_id] = {
                "models": [model_to_dict(m) for m in entry.models],
                "last_modified": entry.last_modified,
                "checked_at": entry.checked_at,
                "etag": entry.etag,
            }
            self._save(data)

    async def delete(self, provider_id: str) -> None:
        with self._file_lock:
            data = self._load()
            if provider_id in data:
                del data[provider_id]
                self._save(data)


__all__ = [
    "ModelsStoreEntry",
    "ModelsStore",
    "ProviderModelsStore",
    "provider_models_store",
    "InMemoryModelsStore",
    "FileModelsStore",
    "model_to_dict",
    "model_from_dict",
]
