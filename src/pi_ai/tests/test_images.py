"""图片生成运行时（注册表 / ImagesModels 集合）测试。"""

import pytest

import pi_ai.images as images_api
from pi_ai.images_api_registry import (
    clear_images_api_providers,
    get_images_api_provider,
    register_images_api_provider,
    unregister_images_api_providers,
)
from pi_ai.images_models import create_images_models, create_images_provider
from pi_ai.auth import env_api_key_auth
from pi_ai.auth.credential_store import InMemoryCredentialStore
from pi_ai.types import ImagesContext, ImagesModel


def _image_model() -> ImagesModel:
    return ImagesModel(
        id="openai/gpt-image-1",
        api="openrouter-images",
        provider="openrouter",
        input=["text", "image"],
        output=["text", "image"],
    )


def _context() -> ImagesContext:
    return {"input": [{"type": "text", "text": "draw a cat"}]}


async def _fake_generate(model, context, options=None):
    return {
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "output": [{"type": "text", "text": "done"}],
        "stop_reason": "stop",
        "timestamp": 1,
    }


@pytest.mark.asyncio
async def test_registry_register_and_dispatch():
    clear_images_api_providers()
    register_images_api_provider("openrouter-images", _fake_generate, source_id="test")
    provider_fn = get_images_api_provider("openrouter-images")
    assert provider_fn is not None
    result = await provider_fn(_image_model(), _context())
    assert result["stop_reason"] == "stop"
    # api 不匹配抛错
    mismatched = ImagesModel(id="x", api="other", provider="p")
    with pytest.raises(ValueError, match="Mismatched api"):
        await provider_fn(mismatched, _context())
    unregister_images_api_providers("test")
    assert get_images_api_provider("openrouter-images") is None


def _provider():
    return create_images_provider(
        id="openrouter",
        name="OpenRouter",
        auth=env_api_key_auth("OpenRouter", ["OPENROUTER_API_KEY"]),
        models=[_image_model()],
        api=_fake_generate,
    )


@pytest.mark.asyncio
async def test_images_models_basic():
    models = create_images_models()
    models.set_provider(_provider())
    assert models.get_provider("openrouter") is not None
    assert [m.id for m in models.get_models()] == ["openai/gpt-image-1"]
    assert models.get_model("openrouter", "openai/gpt-image-1") is not None
    result = await models.generate_images(_image_model(), _context())
    assert result["stop_reason"] == "stop"


@pytest.mark.asyncio
async def test_images_models_generate_never_rejects():
    models = create_images_models()
    result = await models.generate_images(_image_model(), _context())
    assert result["stop_reason"] == "error"
    assert "Unknown provider" in result["error_message"]


@pytest.mark.asyncio
async def test_images_models_get_auth_env(monkeypatch):
    models = create_images_models()
    models.set_provider(_provider())
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    result = await models.get_auth("openrouter")
    assert result is not None
    assert result.auth["api_key"] == "sk-or"


@pytest.mark.asyncio
async def test_images_models_refresh_single_failure():
    async def bad_refresh():
        raise RuntimeError("network down")

    provider = create_images_provider(
        id="dyn",
        name="Dynamic",
        auth=None,
        models=[],
        api=_fake_generate,
        refresh_models=bad_refresh,
    )
    models = create_images_models()
    models.set_provider(provider)
    with pytest.raises(Exception, match="network down"):
        await models.refresh("dyn")


@pytest.mark.asyncio
async def test_images_models_refresh_all_best_effort():
    async def bad_refresh():
        raise RuntimeError("network down")

    provider = create_images_provider(
        id="dyn",
        name="Dynamic",
        auth=None,
        models=[],
        api=_fake_generate,
        refresh_models=bad_refresh,
    )
    models = create_images_models()
    models.set_provider(provider)
    # 全量刷新 best-effort：不抛异常。
    await models.refresh()


@pytest.mark.asyncio
async def test_images_models_stores_oauth_refresh_path():
    """OAuth 凭证经 resolve_provider_auth 自动刷新（复用 M4 路径）。"""
    from pi_ai.types import now_ms

    class _OAuthAuth:
        name = "Fake"

        async def refresh(self, credential, signal=None):
            return {**credential, "access": "rotated", "expires": now_ms() + 3600_000}

        async def to_auth(self, credential):
            return {"api_key": credential["access"]}

    class _OAuthProvider:
        id = "openrouter"
        auth = type("Auth", (), {"oauth": _OAuthAuth()})()

    store = InMemoryCredentialStore()
    await store.write(
        "openrouter",
        {"type": "oauth", "access": "old", "refresh": "r", "expires": now_ms() - 1000},
    )
    models = create_images_models(credentials=store)
    models.set_provider(_OAuthProvider())  # type: ignore[arg-type]
    result = await models.get_auth("openrouter")
    assert result is not None
    assert result.source == "OAuth"
    assert result.auth["api_key"] == "rotated"


@pytest.mark.asyncio
async def test_generate_images_unregistered_api_raises():
    clear_images_api_providers()
    model = ImagesModel(id="x", api="missing", provider="p")
    with pytest.raises(ValueError, match="No API provider registered"):
        await images_api.generate_images(model, _context())


def test_register_builtin_images_api_providers_idempotent():
    from pi_ai.providers import images as images_module

    clear_images_api_providers()
    images_module._registered = False

    images_module.register_builtin_images_api_providers()
    images_module.register_builtin_images_api_providers()
    assert get_images_api_provider("openrouter-images") is not None
    unregister_images_api_providers("builtin")
    assert get_images_api_provider("openrouter-images") is None
