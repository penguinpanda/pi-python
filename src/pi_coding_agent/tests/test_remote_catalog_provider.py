"""远程目录 overlay 测试。"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from pi_ai.models.models_store import InMemoryModelsStore, ModelsStoreEntry
from pi_ai.provider import Provider, RefreshModelsContext, create_provider
from pi_ai.types import Model
from pi_coding_agent.remote_catalog_provider import (
    _merge_models,
    _parse_catalog,
    with_remote_catalog,
)


def _provider() -> Provider:
    return create_provider(
        id="openai",
        name="OpenAI",
        auth=None,
        models=[
            Model(id="gpt-base", provider="openai", api="responses"),
        ],
        api_kind="responses",
    )


def test_merge_models_overrides_same_id() -> None:
    baseline = [Model(id="a", provider="p", api="completions")]
    dynamic = [
        Model(id="a", provider="p", api="completions", max_tokens=999),
        Model(id="b", provider="p", api="completions"),
    ]
    merged = _merge_models(baseline, dynamic)
    assert [m.id for m in merged] == ["a", "b"]
    assert merged[0].max_tokens == 999


def test_parse_catalog_list_and_map_forms() -> None:
    models = _parse_catalog("openai", [{"id": "gpt-x", "name": "X"}])
    assert models[0].id == "gpt-x"
    assert models[0].provider == "openai"
    assert models[0].api == "completions"
    models2 = _parse_catalog("openai", {"models": [{"id": "gpt-y"}]})
    assert models2[0].id == "gpt-y"
    with pytest.raises(RuntimeError):
        _parse_catalog("openai", "nonsense")


@pytest.mark.asyncio
async def test_remote_catalog_refresh_and_304_window(monkeypatch) -> None:
    import pi_coding_agent.remote_catalog_provider as mod
    from pi_ai.models.models_store import provider_models_store

    provider = _provider()
    overlaid = with_remote_catalog(provider, "https://cat.example.com")
    store = provider_models_store(InMemoryModelsStore(), provider.id)

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content=json.dumps([{"id": "gpt-remote"}]),
            headers={"etag": '"v1"', "last-modified": "Wed, 01 Jan 2025 00:00:00 GMT"},
        )

    monkeypatch.setattr(
        mod,
        "_client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )

    context = RefreshModelsContext(store=store, allow_network=True, force=True)
    await overlaid.refresh_models(context)

    models = overlaid.get_models()
    assert "gpt-remote" in [m.id for m in models]
    assert "if-none-match" not in captured["headers"]  # 首次请求无校验器
    entry = await store.read()
    assert entry is not None
    assert entry.etag == '"v1"'
    assert entry.checked_at is not None

    # 新鲜度窗口内（4h）不重验。
    captured.clear()
    context2 = RefreshModelsContext(store=store, allow_network=True, force=False)
    await overlaid.refresh_models(context2)
    assert captured == {}

    # 窗口过期后重验：携带 If-None-Match；304 只更新 checkedAt。
    entry_old = await store.read()
    assert entry_old is not None
    await store.write(
        ModelsStoreEntry(
            models=entry_old.models,
            checked_at=int(time.time() * 1000) - 5 * 60 * 60 * 1000,
            last_modified=1,
            etag='"v1"',
        ),
    )

    def handler_304(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(304)

    monkeypatch.setattr(
        mod,
        "_client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler_304), **kw),
    )
    context3 = RefreshModelsContext(store=store, allow_network=True, force=False)
    await overlaid.refresh_models(context3)
    assert captured["headers"].get("if-none-match") == '"v1"'
    entry_after = await store.read()
    assert entry_after is not None
    assert entry_after.etag == '"v1"'
    # 304 后 overlay 仍保持
    assert "gpt-remote" in [m.id for m in overlaid.get_models()]
