"""ModelsStore / 序列化测试。"""

import pytest

from pi_ai.models.models_store import (
    FileModelsStore,
    InMemoryModelsStore,
    ModelsStoreEntry,
    model_from_dict,
    model_to_dict,
    provider_models_store,
)
from pi_ai.types import Model, ModelCost, ModelCostTier


def _model(model_id: str = "gpt-test") -> Model:
    return Model(
        id=model_id,
        provider="openai",
        api="openai-responses",
        name="GPT Test",
        input=["text", "image"],
        output=["text"],
        cost=ModelCost(
            input=1.0,
            output=2.0,
            cache_read=0.1,
            cache_write=0.2,
            tiers=[ModelCostTier(input=0.5, output=1.0, input_tokens_above=1000)],
        ),
        max_tokens=8192,
        base_url="https://api.openai.com/v1",
        context_window=128000,
        compat={"supportsStrictMode": True},
        thinking_level_map={"high": "high"},
        reasoning=True,
    )


def test_model_serialization_roundtrip():
    model = _model()
    restored = model_from_dict(model_to_dict(model))
    assert restored == model
    assert restored.cost.tiers[0].input_tokens_above == 1000
    assert "image" in restored.input
    assert restored.reasoning is True


@pytest.mark.asyncio
async def test_in_memory_store_isolation():
    store = InMemoryModelsStore()
    a = provider_models_store(store, "a")
    provider_models_store(store, "b")
    await a.write(ModelsStoreEntry(models=[_model("m-a")]))
    assert await store.read("a") is not None
    assert await store.read("b") is None
    await a.delete()
    assert await store.read("a") is None


@pytest.mark.asyncio
async def test_in_memory_store_returns_copies():
    store = InMemoryModelsStore()
    await store.write("a", ModelsStoreEntry(models=[_model()]))
    entry = await store.read("a")
    entry.models[0].id = "mutated"
    fresh = await store.read("a")
    assert fresh.models[0].id == "gpt-test"


@pytest.mark.asyncio
async def test_file_store_persistence(tmp_path):
    path = tmp_path / "models.json"
    store = FileModelsStore(path)
    await store.write("openai", ModelsStoreEntry(models=[_model()], etag='"abc"', checked_at=1))
    assert path.exists()
    loaded = await store.read("openai")
    assert loaded is not None
    assert loaded.models[0].id == "gpt-test"
    assert loaded.etag == '"abc"'
    assert loaded.checked_at == 1
    await store.delete("openai")
    assert await store.read("openai") is None


@pytest.mark.asyncio
async def test_file_store_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "models.json"
    path.write_text("not json")
    store = FileModelsStore(path)
    assert await store.read("openai") is None
