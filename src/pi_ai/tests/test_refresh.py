"""动态模型刷新（Provider.refresh_models / Models.refresh）测试。"""

import asyncio

import pytest

from pi_ai import Models
from pi_ai.models import ModelsRefreshOptions
from pi_ai.models.models_store import (
    InMemoryModelsStore,
    ModelsStoreEntry,
    provider_models_store,
)
from pi_ai.provider import RefreshModelsContext, create_provider
from pi_ai.types import Model


def _model(model_id: str) -> Model:
    return Model(id=model_id, provider="dyn", api="openai-completions")


@pytest.fixture
def dynamic_provider():
    async def fetch(context: RefreshModelsContext) -> list[Model]:
        return [_model("dyn-1"), _model("dyn-2")]

    provider = create_provider(
        id="dyn",
        name="Dynamic",
        auth=None,
        models=[_model("static-0")],
        api_kind="completions",
        fetch_models=fetch,
    )
    return provider


@pytest.mark.asyncio
async def test_refresh_models_fetches_and_merges(dynamic_provider):
    store = InMemoryModelsStore()
    result = await dynamic_provider.refresh_models(
        RefreshModelsContext(
            store=provider_models_store(store, "dyn"),
            allow_network=True,
        )
    )
    assert result is None
    ids = [m.id for m in dynamic_provider.get_models()]
    assert "static-0" in ids
    assert "dyn-1" in ids
    assert "dyn-2" in ids
    entry = await store.read("dyn")
    assert entry is not None
    assert entry.checked_at is not None


@pytest.mark.asyncio
async def test_refresh_models_offline_restores_cache(dynamic_provider):
    store = InMemoryModelsStore()
    await store.write("dyn", ModelsStoreEntry(models=[_model("cached-1")]))
    await dynamic_provider.refresh_models(
        RefreshModelsContext(
            store=provider_models_store(store, "dyn"),
            allow_network=False,
        )
    )
    ids = [m.id for m in dynamic_provider.get_models()]
    assert "cached-1" in ids
    assert "dyn-1" not in ids  # 离线时不应抓取


@pytest.mark.asyncio
async def test_refresh_models_concurrent_shared_inflight(dynamic_provider):
    fetch_calls = 0

    async def fetch(context):
        nonlocal fetch_calls
        fetch_calls += 1
        await asyncio.sleep(0.01)
        return [_model("dyn-1")]

    provider = create_provider(id="dyn", name="Dynamic", auth=None, models=[], fetch_models=fetch)
    store = InMemoryModelsStore()
    ctx = RefreshModelsContext(store=provider_models_store(store, "dyn"), allow_network=True)
    await asyncio.gather(provider.refresh_models(ctx), provider.refresh_models(ctx))
    assert fetch_calls == 1


@pytest.mark.asyncio
async def test_models_refresh_collects_errors(dynamic_provider):
    async def fetch(context):
        raise RuntimeError("boom")

    provider = create_provider(
        id="dyn", name="Dynamic", auth=None, models=[_model("static")], fetch_models=fetch
    )
    models = Models(models_store=InMemoryModelsStore())
    models.add_provider(provider)
    result = await models.refresh()
    assert "dyn" in result.errors
    assert "boom" in str(result.errors["dyn"])
    # 失败后保留静态列表（best-effort 缓存恢复不覆盖）。
    assert [m.id for m in provider.get_models()] == ["static"]


@pytest.mark.asyncio
async def test_models_refresh_abort_signal(dynamic_provider):
    signal = asyncio.Event()
    signal.set()
    models = Models(models_store=InMemoryModelsStore())
    models.add_provider(dynamic_provider)
    result = await models.refresh(ModelsRefreshOptions(signal=signal))
    assert result.aborted is True
    assert result.errors == {}


def test_ollama_provider_has_refresh_models():
    from pi_ai.providers.ollama import ollama_provider

    provider = ollama_provider()
    assert provider.refresh_models is not None
