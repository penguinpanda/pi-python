"""Radius provider 测试。"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai.providers.all import create_default_models
from pi_ai.providers.radius import (
    _sanitize_gateway_config,
    _fetch_radius_models,
    get_radius_models_from_config,
    normalize_radius_gateway_url,
    radius_provider,
)
from pi_ai.provider import RefreshModelsContext

_VALID_CONFIG = {
    "baseUrl": "https://gw.example.com",
    "models": [
        {
            "id": "r1",
            "name": "Radius One",
            "reasoning": True,
            "input": ["text", "image"],
            "cost": {"input": 1.0, "output": 2.0},
            "contextWindow": 200000,
            "maxTokens": 16000,
        },
        {
            "id": "bad",
            "name": "bad",
            "reasoning": "not-bool",
            "input": [],
            "cost": {},
            "contextWindow": "x",
            "maxTokens": 0,
        },
    ],
}


def test_radius_provider_registered() -> None:
    provider = create_default_models().get_provider("radius")
    assert provider is not None
    assert provider.name == "Radius"
    assert provider.id == "radius"


def test_normalize_gateway_url() -> None:
    assert normalize_radius_gateway_url("radius.pi.dev") == "https://radius.pi.dev"
    assert normalize_radius_gateway_url("https://x.dev/") == "https://x.dev"


def test_sanitize_gateway_config() -> None:
    config = _sanitize_gateway_config(_VALID_CONFIG)
    assert config is not None
    assert config["baseUrl"] == "https://gw.example.com"
    assert len(config["models"]) == 1  # 非法条目被过滤


def test_get_radius_models_from_config() -> None:
    config = _sanitize_gateway_config(_VALID_CONFIG)
    assert config is not None
    models = get_radius_models_from_config("radius", config)
    assert len(models) == 1
    model = models[0]
    assert model.id == "r1"
    assert model.api == "pi-messages"
    assert model.provider == "radius"
    assert model.base_url == "https://gw.example.com"
    assert model.reasoning is True
    assert model.cost.input == 1.0


@pytest.mark.asyncio
async def test_fetch_radius_models_from_gateway(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/config"
        return httpx.Response(200, content=json.dumps(_VALID_CONFIG).encode())

    monkeypatch.setattr(
        "pi_ai.providers.radius._client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )
    context = RefreshModelsContext(
        allow_network=True,
        credential={"key": "rk"},
    )
    models = await _fetch_radius_models("radius", "https://gw.example.com", context)
    assert len(models) == 1
    assert models[0].id == "r1"


@pytest.mark.asyncio
async def test_fetch_radius_models_offline_returns_empty(monkeypatch) -> None:
    async def fail_handler(*args, **kwargs):
        raise AssertionError("should not hit network")

    monkeypatch.setattr(
        "pi_ai.providers.radius._client_factory",
        lambda *a, **kw: fail_handler(),  # type: ignore[misc]
    )
    context = RefreshModelsContext(allow_network=False)
    models = await _fetch_radius_models("radius", "https://gw.example.com", context)
    assert models == []


@pytest.mark.asyncio
async def test_fetch_radius_models_http_error_wrapped(monkeypatch) -> None:
    """网关请求失败：RuntimeError 包装（含网关信息）。"""
    import httpx
    import pi_ai.providers.radius as radius_mod

    def failing_factory(*a, **kw):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                raise httpx.ConnectError("refused")

        return _Client()

    monkeypatch.setattr(radius_mod, "_client_factory", failing_factory)
    context = RefreshModelsContext(allow_network=True)
    with pytest.raises(RuntimeError, match="Could not load Radius config"):
        await radius_mod._fetch_radius_models("radius", "https://gw.example.com", context)


@pytest.mark.asyncio
async def test_fetch_radius_models_invalid_config(monkeypatch) -> None:
    """网关返回非法配置：RuntimeError。"""
    import json
    import httpx
    import pi_ai.providers.radius as radius_mod

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"not": "valid"}).encode())

    monkeypatch.setattr(
        radius_mod,
        "_client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )
    context = RefreshModelsContext(allow_network=True)
    with pytest.raises(RuntimeError, match="Invalid Radius config"):
        await radius_mod._fetch_radius_models("radius", "https://gw.example.com", context)


@pytest.mark.asyncio
async def test_fetch_radius_models_abort_signal(monkeypatch) -> None:
    """取消信号置位：不发起请求直接返回空。"""
    import asyncio
    import pi_ai.providers.radius as radius_mod

    def failing_factory(*a, **kw):
        raise AssertionError("should not be called")

    monkeypatch.setattr(radius_mod, "_client_factory", failing_factory)
    signal = asyncio.Event()
    signal.set()
    context = RefreshModelsContext(allow_network=True, signal=signal)
    assert await radius_mod._fetch_radius_models("radius", "https://gw.example.com", context) == []


def test_radius_provider_custom_gateway() -> None:
    provider = radius_provider(gateway="custom.example.com")
    assert provider.name == "Radius"
