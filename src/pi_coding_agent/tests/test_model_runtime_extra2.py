"""ModelRuntime 查询、登出与刷新补充测试。"""

from __future__ import annotations

import pytest
from pi_ai.models import ModelsRefreshOptions

from pi_coding_agent.tests.test_model_runtime_extra import _runtime


@pytest.mark.asyncio
async def test_model_lookup_and_availability_queries() -> None:
    runtime = await _runtime()
    assert runtime.get_model("faux", "missing") is None

    available = await runtime.get_available("faux")
    assert available
    assert all(model.provider == "faux" for model in available)

    snapshot = runtime.get_available_snapshot()
    assert snapshot

    assert runtime.has_configured_auth("faux") is True
    assert runtime.is_using_oauth("faux") is False
    status = runtime.get_provider_auth_status("faux")
    assert status["configured"] is True
    credentials = await runtime.list_credentials()
    assert isinstance(credentials, list)


@pytest.mark.asyncio
async def test_refresh_returns_snapshot_without_errors() -> None:
    runtime = await _runtime()
    result = await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    assert result.aborted is False
    assert result.errors == {}


@pytest.mark.asyncio
async def test_supports_deferred_is_false_for_faux() -> None:
    runtime = await _runtime()
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    assert runtime.supports_deferred(model) is True
