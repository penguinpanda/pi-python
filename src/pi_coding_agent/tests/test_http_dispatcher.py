"""HTTP dispatcher 测试。"""

from __future__ import annotations

import httpx

from pi_coding_agent.http_dispatcher import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    apply_http_proxy_settings,
    configure_http_dispatcher,
    make_http_client,
    parse_http_idle_timeout_ms,
    reset_http_dispatcher,
)


def test_parse_and_defaults() -> None:
    assert parse_http_idle_timeout_ms("disabled") == 0
    assert parse_http_idle_timeout_ms("bad") is None
    assert DEFAULT_HTTP_IDLE_TIMEOUT_MS == 300_000


def test_proxy_and_client_factory(monkeypatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    try:
        apply_http_proxy_settings("http://proxy.local:8080")
        configure_http_dispatcher(30_000)
        client = make_http_client()
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.connect == 30
        default_client = httpx.AsyncClient()
        assert default_client.timeout.connect == 30
        limited_client = httpx.AsyncClient(timeout=180)
        assert limited_client.timeout.connect == 30
        assert limited_client.timeout.read == 30
    finally:
        reset_http_dispatcher()
