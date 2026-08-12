"""ModelRuntime 补充测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pi_ai import Context, Model, Models, UserMessage
from pi_ai.auth import ApiKeyCredential, ResolvedAuth
from pi_ai.models import ModelsRefreshOptions
from pi_ai.providers.faux import faux_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import (
    ComposedApiKeyAuth,
    ModelRuntime,
    _validate_extension_provider,
)


async def _runtime() -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    models.add_provider(faux_provider().provider)
    runtime = ModelRuntime(models, store)
    await runtime.get_available()
    return runtime


def test_composed_api_key_auth_branches() -> None:
    base = SimpleNamespace(
        resolve=lambda _credential: ResolvedAuth(api_key="base-key", source="env")
    )
    auth = ComposedApiKeyAuth("p", base, None, [])
    assert auth.resolve(ApiKeyCredential(key="stored")).source == "stored credential"
    assert auth.resolve(None).api_key == "base-key"

    configured = ComposedApiKeyAuth("p", None, "sk-plain", [])
    resolved = configured.resolve(None)
    assert resolved is not None
    assert resolved.source == "configured API key"

    missing = ComposedApiKeyAuth("p", None, "${NOT_SET}", [])
    assert missing.resolve(None) is None


def test_validate_extension_provider_stream_simple_requires_api() -> None:
    with pytest.raises(ValueError, match='"api" is required'):
        _validate_extension_provider("p", None, None, {"stream_simple": lambda *a: None})


@pytest.mark.asyncio
async def test_register_native_provider_empty_id_raises() -> None:
    runtime = await _runtime()
    with pytest.raises(ValueError, match="must not be empty"):
        runtime.register_native_provider(SimpleNamespace(id=" "))


@pytest.mark.asyncio
async def test_compose_model_provider_unknown_raises() -> None:
    runtime = await _runtime()
    with pytest.raises(ValueError, match="Unknown provider"):
        runtime.compose_model_provider("unknown")


@pytest.mark.asyncio
async def test_check_auth_unknown_provider() -> None:
    runtime = await _runtime()
    assert await runtime.check_auth("unknown") is None


@pytest.mark.asyncio
async def test_get_auth_unknown_targets() -> None:
    runtime = await _runtime()
    assert await runtime.get_auth("unknown") is None
    assert await runtime.get_auth(Model(id="m", provider="unknown", api="a")) is None


@pytest.mark.asyncio
async def test_get_auth_model_headers_merge() -> None:
    runtime = await _runtime()
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    model.headers = {"X-Model": "1"}
    result = await runtime.get_auth(model)
    assert result is not None
    assert result.auth["headers"]["X-Model"] == "1"


@pytest.mark.asyncio
async def test_provider_auth_status_and_logout() -> None:
    runtime = await _runtime()
    status = runtime.get_provider_auth_status("faux")
    assert status["configured"] is True

    async def _set(_current):
        return ApiKeyCredential(key="sk")

    await runtime._credentials.modify("faux", _set)
    runtime._stored_providers.add("faux")
    status = runtime.get_provider_auth_status("faux")
    assert status["source"] == "stored"

    await runtime.logout("faux")
    assert await runtime._credentials.read("faux") is None


@pytest.mark.asyncio
async def test_refresh_availability_error() -> None:
    runtime = await _runtime()

    async def _boom():
        raise RuntimeError("availability failed")

    runtime._run_availability_refresh = _boom  # type: ignore[method-assign]
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    assert "availability failed" in runtime.get_error()


@pytest.mark.asyncio
async def test_registered_provider_queries() -> None:
    runtime = await _runtime()
    runtime.register_provider("acme", {"api_key": "sk", "models": []})
    assert "acme" in runtime.get_registered_provider_ids()
    assert runtime.get_registered_provider_config("acme") == {"api_key": "sk", "models": []}
    assert runtime.get_registered_native_provider("faux") is None


@pytest.mark.asyncio
async def test_compatibility_request_config() -> None:
    runtime = await _runtime()
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    model.headers = {"X": "1"}
    config = runtime.get_compatibility_request_config(model)
    assert config["headers"]["X"] == "1"


@pytest.mark.asyncio
async def test_provisional_auth_check() -> None:
    runtime = await _runtime()
    assert runtime._provisional_auth_check("faux", None) is not None
    runtime._stored_providers.add("faux")
    assert runtime._provisional_auth_check("faux", None)["source"] == "stored credential"
    assert runtime._provisional_auth_check("acme", {"api_key": "plain"}) is not None
    assert runtime._provisional_auth_check("acme", {"api_key": "${NOT_SET}"}) is None


@pytest.mark.asyncio
async def test_complete_through_runtime() -> None:
    core = faux_provider()
    core.set_responses([{"role": "assistant", "content": [{"type": "text", "text": "extra"}]}])
    runtime = ModelRuntime(Models(credentials=AuthStorage.in_memory()), AuthStorage.in_memory())
    runtime._models.add_provider(core.provider)
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    context = Context(messages=[UserMessage(role="user", content="hi")])
    message = await runtime.complete(model, context)
    assert message["content"][0]["text"] == "extra"
