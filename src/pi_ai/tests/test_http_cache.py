"""HTTP 缓存语义（etag / last-modified / 304）测试。"""

from pi_ai.models.http_cache import (
    cache_headers,
    extract_cache_metadata,
    http_date,
    parse_http_date,
    updated_entry,
)
from pi_ai.models.models_store import ModelsStoreEntry


def test_cache_headers_from_entry():
    entry = ModelsStoreEntry(models=[], etag='"v1"', last_modified=1_700_000_000_000)
    headers = cache_headers(entry)
    assert headers["If-None-Match"] == '"v1"'
    assert "If-Modified-Since" in headers


def test_cache_headers_none_entry():
    assert cache_headers(None) == {}


def test_http_date_roundtrip():
    ms = 1_700_000_000_000
    assert parse_http_date(http_date(ms)) == ms


def test_parse_http_date_invalid():
    assert parse_http_date(None) is None
    assert parse_http_date("garbage") is None


def test_extract_cache_metadata():
    headers = {"etag": '"xyz"', "last-modified": http_date(1_700_000_000_000)}
    etag, last_modified = extract_cache_metadata(headers)
    assert etag == '"xyz"'
    assert last_modified == 1_700_000_000_000


def test_updated_entry_keeps_previous_metadata_when_missing():
    previous = ModelsStoreEntry(models=[], etag='"old"', last_modified=123)
    entry = updated_entry(previous, [1, 2], {}, checked_at_ms=456)
    assert entry.etag == '"old"'
    assert entry.last_modified == 123
    assert entry.checked_at == 456
    assert entry.models == [1, 2]


def test_updated_entry_takes_new_metadata():
    previous = ModelsStoreEntry(models=[], etag='"old"', last_modified=123)
    headers = {"ETag": '"new"', "Last-Modified": http_date(1_700_000_000_000)}
    entry = updated_entry(previous, [], headers)
    assert entry.etag == '"new"'
    assert entry.last_modified == 1_700_000_000_000
