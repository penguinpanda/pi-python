"""动态模型刷新（Provider.refresh_models / Models.refresh）测试。"""

import asyncio

import pytest
from types import SimpleNamespace

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
async def test_refresh_models_mismatched_context_not_merged():
    """语义不同的并发调用（force 不同）不得共享同一 in-flight 任务。"""
    fetch_calls = 0

    async def fetch(context):
        nonlocal fetch_calls
        fetch_calls += 1
        await asyncio.sleep(0.03)
        return [_model("dyn-1")]

    provider = create_provider(id="dyn", name="Dynamic", auth=None, models=[], fetch_models=fetch)
    store = InMemoryModelsStore()
    forced = RefreshModelsContext(
        store=provider_models_store(store, "dyn"),
        allow_network=True,
        force=True,
    )
    normal = RefreshModelsContext(
        store=provider_models_store(store, "dyn"),
        allow_network=True,
        force=False,
    )
    await asyncio.gather(provider.refresh_models(forced), provider.refresh_models(normal))
    # force 语义不同：后到调用按自己的 context 串行执行，而不是被吸收。
    assert fetch_calls == 2


@pytest.mark.asyncio
async def test_refresh_mismatch_caller_does_not_inherit_failure():
    """语义不同的调用者不得继承 in-flight 任务的异常（离线恢复不被跳过）。"""
    calls: list[bool] = []

    async def fetch(context):
        calls.append(context.force)
        if context.force:
            await asyncio.sleep(0.03)
            raise RuntimeError("force fetch failed")
        return [_model("dyn-1")]

    provider = create_provider(id="dyn", name="Dynamic", auth=None, models=[], fetch_models=fetch)
    store = InMemoryModelsStore()
    forced = RefreshModelsContext(
        store=provider_models_store(store, "dyn"),
        allow_network=True,
        force=True,
    )
    normal = RefreshModelsContext(
        store=provider_models_store(store, "dyn"),
        allow_network=True,
        force=False,
    )

    first = asyncio.create_task(provider.refresh_models(forced))
    await asyncio.sleep(0.005)
    # 后到的正常调用不得继承 force 任务的异常，应按自己的 context 执行。
    await provider.refresh_models(normal)
    try:
        await first
    except RuntimeError:
        pass
    assert calls == [True, False]
    assert [m.id for m in provider.get_models()] == ["dyn-1"]


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


@pytest.mark.asyncio
async def test_refresh_empty_fetch_keeps_cached_store():
    """空抓取结果不得覆盖非空缓存（认证缺失时防目录清空）。"""
    cached = [_model("cached-1"), _model("cached-2")]
    store = InMemoryModelsStore()
    await store.write("dyn", ModelsStoreEntry(models=cached))

    async def fetch(context):
        return []

    provider = create_provider(
        id="dyn", name="Dynamic", auth=None, models=[_model("static-0")], fetch_models=fetch
    )
    await provider.refresh_models(
        RefreshModelsContext(store=provider_models_store(store, "dyn"), allow_network=True)
    )
    ids = [m.id for m in provider.get_models()]
    assert "cached-1" in ids and "cached-2" in ids
    entry = await store.read("dyn")
    assert entry is not None
    assert [m.id for m in entry.models] == ["cached-1", "cached-2"]


@pytest.mark.asyncio
async def test_models_refresh_resolves_env_credential(monkeypatch):
    """仅环境变量认证时，refresh 也应解析 env key 传给 fetch。"""
    from pi_ai.auth import env_api_key_auth

    seen: list[object] = []

    async def fetch(context):
        seen.append(context.credential)
        return [_model("dyn-1")]

    monkeypatch.setenv("PI_TEST_REVIEW_API_KEY", "sk-env-key")
    provider = create_provider(
        id="dyn",
        name="Dynamic",
        auth=env_api_key_auth("Dynamic", ["PI_TEST_REVIEW_API_KEY"]),
        models=[],
        fetch_models=fetch,
    )
    models = Models(models_store=InMemoryModelsStore())
    models.add_provider(provider)
    result = await models.refresh()
    assert result.errors == {}
    assert seen == [{"type": "api_key", "key": "sk-env-key"}]


@pytest.mark.asyncio
async def test_models_refresh_refreshes_oauth_credential():
    """目录抓取前刷新过期 OAuth token（对齐 Provider.stream 路径）。"""
    seen: dict[str, object] = {}

    async def fetch(context):
        seen["credential"] = context.credential
        return [_model("dyn-1")]

    class _FakeOAuth:
        async def refresh(self, credential, signal=None):
            return {**credential, "access": "refreshed-access"}

        async def to_auth(self, credential):
            return {"api_key": credential["access"]}

    provider = create_provider(
        id="dyn",
        name="Dynamic",
        auth=SimpleNamespace(oauth=_FakeOAuth()),
        models=[],
        fetch_models=fetch,
    )
    models = Models(models_store=InMemoryModelsStore())
    models.add_provider(provider)
    await models._credentials.write(
        "dyn",
        {"type": "oauth", "access": "old-access", "refresh": "r", "expires": 1},
    )
    result = await models.refresh()
    assert result.errors == {}
    assert seen["credential"]["access"] == "refreshed-access"


@pytest.mark.asyncio
async def test_refresh_caller_cancel_keeps_inflight_dedup():
    """调用方被取消后：无孤儿任务、任意时刻 fetch 并发不超过 1、目录最终正确。

    Python 3.11+ 取消会传播给共享任务（任务随之终止、引用由 done 回调清理）；
    3.10 不会传播（任务继续运行，新调用合并进同一 in-flight）。两种语义下
    fetch 都不能并发执行，且第二次刷新必须成功落地。
    """
    active = 0
    max_active = 0
    fetch_calls = 0

    async def fetch(context):
        nonlocal active, max_active, fetch_calls
        active += 1
        max_active = max(max_active, active)
        fetch_calls += 1
        try:
            await asyncio.sleep(0.05)
            return [_model("dyn-1")]
        finally:
            active -= 1

    provider = create_provider(id="dyn", name="Dynamic", auth=None, models=[], fetch_models=fetch)
    ctx = RefreshModelsContext(
        store=provider_models_store(InMemoryModelsStore(), "dyn"),
        allow_network=True,
    )

    first = asyncio.create_task(provider.refresh_models(ctx))
    await asyncio.sleep(0.005)
    first.cancel()
    try:
        await first
    except asyncio.CancelledError:
        pass

    await provider.refresh_models(ctx)
    assert max_active == 1
    assert [m.id for m in provider.get_models()] == ["dyn-1"]
    assert fetch_calls >= 1


def test_ollama_provider_has_refresh_models():
    from pi_ai.providers.ollama import ollama_provider

    provider = ollama_provider()
    assert provider.refresh_models is not None
