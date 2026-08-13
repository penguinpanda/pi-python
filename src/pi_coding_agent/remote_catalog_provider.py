"""远程模型目录 overlay（对齐 TS core/remote-catalog-provider.ts）。

为静态内置 provider 叠加 pi.dev 持久化目录：ETag/Last-Modified 条件请求，
4 小时新鲜度窗口，304/404/501 与瞬时失败均有明确语义。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any
from urllib.parse import quote

import httpx

from pi_ai.models.models_store import ModelsStoreEntry, model_from_dict
from pi_ai.provider import Provider, RefreshModelsContext
from pi_ai.types import Model

DEFAULT_CATALOG_BASE_URL = "https://pi.dev"
REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000

_UserAgent = "pi-python/0.1.0"

# 测试可替换的 client 工厂（间接层避免 monkeypatch httpx.AsyncClient 全局递归）。
_client_factory = httpx.AsyncClient


def _now_ms() -> int:
    return int(time.time() * 1000)


def _merge_models(baseline: list[Model], dynamic: list[Model]) -> list[Model]:
    """动态目录覆盖同 id 模型，其余追加（对齐 TS mergeModels）。"""
    merged = list(baseline)
    for model in dynamic:
        index = next((i for i, entry in enumerate(merged) if entry.id == model.id), -1)
        if index >= 0:
            merged[index] = model
        else:
            merged.append(model)
    return merged


def _parse_catalog(provider_id: str, value: Any) -> list[Model]:
    """目录 JSON → Model 列表（对齐 TS parseCatalog）。"""
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict) and isinstance(value.get("models"), list):
        entries = value["models"]
    elif isinstance(value, dict):
        entries = list(value.values())
    else:
        raise RuntimeError(f'Invalid model catalog for provider "{provider_id}"')
    models: list[Model] = []
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        entry = {**entry, "provider": provider_id}
        if "api" not in entry:
            entry["api"] = "completions"
        try:
            models.append(model_from_dict(entry))
        except Exception:
            continue
    return models


def _remote_models(entry: ModelsStoreEntry | None, local_generated_at: int | None) -> list[Model]:
    if entry is None:
        return []
    if local_generated_at is not None and (
        entry.last_modified is None or entry.last_modified <= local_generated_at
    ):
        return []
    return list(entry.models)


def with_remote_catalog(
    provider: Provider,
    catalog_base_url: str = DEFAULT_CATALOG_BASE_URL,
    local_generated_at: int | None = None,
) -> Provider:
    """给内置 provider 叠加远程目录（对齐 TS withRemoteCatalog）。"""
    overlaid = replace(provider, refresh_models=None)

    async def refresh(context: RefreshModelsContext) -> None:
        stored = await context.store.read() if context.store is not None else None
        restored = [
            model
            for model in _remote_models(stored, local_generated_at)
            if model.provider == provider.id
        ]
        overlaid._dynamic_models = restored  # 对齐 TS publish update
        if not context.allow_network:
            return
        if context.signal is not None and context.signal.is_set():
            return
        if (
            not context.force
            and stored is not None
            and stored.checked_at is not None
            and stored.last_modified is not None
            and _now_ms() - stored.checked_at < REMOTE_CATALOG_REFRESH_INTERVAL_MS
        ):
            return

        # 仅在有缓存正文时携带校验器：304 不会把 overlay 清空。
        validator = stored.etag if stored is not None and stored.models else None
        url = f"{catalog_base_url}/api/models/providers/{quote(provider.id)}"
        headers: dict[str, str] = {
            "accept": "application/json",
            "User-Agent": _UserAgent,
        }
        if validator:
            headers["if-none-match"] = validator

        try:
            async with _client_factory(timeout=10) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            if context.signal is not None and context.signal.is_set():
                return
            # 连接失败（离线/网络不可达）：同样记录 checkedAt，
            # 4 小时新鲜度窗口内不再重试，避免每次启动阻塞在目录刷新。
            if context.store is not None:
                base = stored or ModelsStoreEntry(
                    models=[], checked_at=None, last_modified=None, etag=None
                )
                await context.store.write(
                    ModelsStoreEntry(
                        models=list(base.models),
                        checked_at=_now_ms(),
                        last_modified=base.last_modified or 0,
                        etag=base.etag,
                    )
                )
            raise RuntimeError(f"Model catalog request failed for {provider.id}: {exc}") from exc
        if context.signal is not None and context.signal.is_set():
            return
        checked_at = _now_ms()

        if response.status_code == 304 and stored is not None:
            if context.store is not None:
                await context.store.write(
                    ModelsStoreEntry(
                        models=list(stored.models),
                        checked_at=checked_at,
                        last_modified=stored.last_modified,
                        etag=stored.etag,
                    )
                )
            return
        if response.status_code in (404, 501):
            # 目录下线：动态模型清空，合并结果中下线模型消失（对齐 TS 语义）。
            overlaid._dynamic_models = []
            if context.store is not None:
                await context.store.write(
                    ModelsStoreEntry(
                        models=[],
                        checked_at=checked_at,
                        last_modified=0,
                        etag=None,
                    )
                )
            return
        if not response.is_success:
            # 瞬时失败：缓存正文与校验器保持有效，保留 etag 下次重验。
            if context.store is not None and stored is not None:
                await context.store.write(
                    ModelsStoreEntry(
                        models=list(stored.models),
                        checked_at=checked_at,
                        last_modified=stored.last_modified,
                        etag=stored.etag,
                    )
                )
            raise RuntimeError(
                f"Model catalog request failed for {provider.id}: {response.status_code}"
            )

        refreshed = _parse_catalog(provider.id, response.json())
        last_modified_raw = response.headers.get("last-modified")
        last_modified = 0
        if last_modified_raw:
            from email.utils import parsedate_to_datetime

            try:
                last_modified = int(parsedate_to_datetime(last_modified_raw).timestamp() * 1000)
            except (TypeError, ValueError):
                last_modified = 0
        if context.signal is not None and context.signal.is_set():
            return
        entry = ModelsStoreEntry(
            models=refreshed,
            checked_at=checked_at,
            last_modified=last_modified,
            etag=response.headers.get("etag"),
        )
        overlaid._dynamic_models = _remote_models(entry, local_generated_at)
        if context.store is not None:
            await context.store.write(entry)

    overlaid.refresh_models = refresh  # type: ignore[assignment]
    return overlaid
