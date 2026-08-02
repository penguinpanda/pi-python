"""动态模型刷新用的 HTTP 缓存语义（etag / last-modified / 304）。"""

from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any

from .models_store import ModelsStoreEntry


def cache_headers(entry: ModelsStoreEntry | None) -> dict[str, str]:
    """从缓存条目组装 If-None-Match / If-Modified-Since。"""
    headers: dict[str, str] = {}
    if entry is None:
        return headers
    if entry.etag:
        headers["If-None-Match"] = entry.etag
    if entry.last_modified is not None:
        headers["If-Modified-Since"] = http_date(entry.last_modified)
    return headers


def http_date(unix_ms: int) -> str:
    """Unix 毫秒时间戳 → HTTP-date。"""
    return format_datetime(
        datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc), usegmt=True
    )


def parse_http_date(value: str | None) -> int | None:
    """HTTP-date → Unix 毫秒时间戳；解析失败返回 None。"""
    if not value:
        return None
    try:
        return int(parsedate_to_datetime(value).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        return None


def extract_cache_metadata(headers: Any) -> tuple[str | None, int | None]:
    """从响应头提取 (etag, last_modified_ms)。"""
    etag = None
    last_modified = None
    try:
        get = headers.get if hasattr(headers, "get") else None
        if get is not None:
            etag = get("etag") or get("ETag") or None
            last_modified = parse_http_date(
                get("last-modified") or get("Last-Modified") or None
            )
    except Exception:
        pass
    return etag, last_modified


def updated_entry(
    previous: ModelsStoreEntry | None,
    models: list[Any],
    response_headers: Any,
    checked_at_ms: int | None = None,
) -> ModelsStoreEntry:
    """构造写入 store 的新条目（保留 304 语义下的 etag/last_modified）。"""
    etag, last_modified = extract_cache_metadata(response_headers)
    return ModelsStoreEntry(
        models=list(models),
        last_modified=last_modified
        if last_modified is not None
        else (previous.last_modified if previous else None),
        checked_at=checked_at_ms,
        etag=etag if etag is not None else (previous.etag if previous else None),
    )


__all__ = [
    "cache_headers",
    "http_date",
    "parse_http_date",
    "extract_cache_metadata",
    "updated_entry",
]
