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
async def test_remote_catalog_404_clears_dynamic_models(monkeypatch) -> None:
    """404/501：动态模型清空，合并结果中下线模型消失。"""
    import pi_coding_agent.remote_catalog_provider as mod
    from pi_ai.models.models_store import InMemoryModelsStore, provider_models_store

    provider = _provider()
    overlaid = with_remote_catalog(provider, "https://cat.example.com")
    store = provider_models_store(InMemoryModelsStore(), provider.id)
    # 先放入缓存目录
    await store.write(
        ModelsStoreEntry(
            models=[Model(id="gpt-stale", provider="openai", api="responses")],
            checked_at=int(time.time() * 1000) - 5 * 60 * 60 * 1000,
            last_modified=1,
            etag='"v1"',
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    monkeypatch.setattr(
        mod,
        "_client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )
    context = RefreshModelsContext(store=store, allow_network=True, force=True)
    await overlaid.refresh_models(context)

    # 动态模型被清空（缓存内容不再返回）
    assert [m.id for m in overlaid.get_models()] == ["gpt-base"]
    entry = await store.read()
    assert entry is not None
    assert entry.models == []
    assert entry.last_modified == 0


@pytest.mark.asyncio
async def test_remote_catalog_transient_failure_keeps_etag(monkeypatch) -> None:
    """瞬时失败：保留 etag 与缓存正文，抛错供上层记录。"""
    import pi_coding_agent.remote_catalog_provider as mod
    from pi_ai.models.models_store import InMemoryModelsStore, provider_models_store

    provider = _provider()
    overlaid = with_remote_catalog(provider, "https://cat.example.com")
    store = provider_models_store(InMemoryModelsStore(), provider.id)
    await store.write(
        ModelsStoreEntry(
            models=[Model(id="gpt-cached", provider="openai", api="responses")],
            checked_at=int(time.time() * 1000) - 5 * 60 * 60 * 1000,
            last_modified=1,
            etag='"v1"',
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    monkeypatch.setattr(
        mod,
        "_client_factory",
        lambda *a, **kw: httpx.AsyncClient(*a, transport=httpx.MockTransport(handler), **kw),
    )
    context = RefreshModelsContext(store=store, allow_network=True, force=True)
    with pytest.raises(RuntimeError):
        await overlaid.refresh_models(context)

    entry = await store.read()
    assert entry is not None
    assert entry.etag == '"v1"'  # 校验器保留，下次重验
    assert [m.id for m in entry.models] == ["gpt-cached"]
    # 缓存恢复的 overlay 仍可用
    assert "gpt-cached" in [m.id for m in overlaid.get_models()]


@pytest.mark.asyncio
async def test_remote_catalog_connect_failure_records_checked_at(monkeypatch) -> None:
    """连接失败：记录 checkedAt（4h 窗口内不再重试），下次启动不阻塞。"""
    import pi_coding_agent.remote_catalog_provider as mod
    from pi_ai.models.models_store import InMemoryModelsStore, provider_models_store

    provider = _provider()
    overlaid = with_remote_catalog(provider, "https://cat.example.com")
    store = provider_models_store(InMemoryModelsStore(), provider.id)

    def failing_factory(*a, **kw):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                import httpx

                raise httpx.ConnectError("unreachable")

        return _Client()

    monkeypatch.setattr(mod, "_client_factory", failing_factory)
    context = RefreshModelsContext(store=store, allow_network=True, force=True)
    with pytest.raises(RuntimeError):
        await overlaid.refresh_models(context)

    entry = await store.read()
    assert entry is not None
    assert entry.checked_at is not None  # 已记录：窗口内不再重试

    # 窗口内第二次 refresh：直接跳过（不发起请求）
    requests = []

    def counting_factory(*a, **kw):
        requests.append(1)
        raise AssertionError("should not be called")

    monkeypatch.setattr(mod, "_client_factory", counting_factory)
    context2 = RefreshModelsContext(store=store, allow_network=True, force=False)
    await overlaid.refresh_models(context2)
    assert requests == []


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
