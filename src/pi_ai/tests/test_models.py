"""
Unit tests for Models registry.
"""

import pytest
from pi_ai import (
    Models,
    Model,
    Context,
    Tool,
    Provider,
    create_provider,
    create_default_models,
    deepseek_provider,
    openai_provider,
)
from pi_ai.auth import env_api_key_auth, InMemoryCredentialStore, ApiKeyCredential, resolve_api_key


class TestModelsRegistry:
    """Provider management and model lookup."""

    def test_create_default_models(self):
        models = create_default_models()
        providers = models.get_providers()
        assert len(providers) == 2
        assert models.get_provider("openai") is not None
        assert models.get_provider("deepseek") is not None

    def test_get_models_all(self):
        models = create_default_models()
        all_models = models.get_models()
        # gpt-4o, gpt-4o-mini, o4-mini, deepseek-chat, deepseek-reasoner, deepseek-v4-flash
        assert len(all_models) == 6

    def test_get_models_by_provider(self):
        models = create_default_models()
        openai_models = models.get_models("openai")
        assert len(openai_models) == 3
        assert any(m.id == "gpt-4o" for m in openai_models)

        deepseek_models = models.get_models("deepseek")
        assert len(deepseek_models) == 3
        assert any(m.id == "deepseek-chat" for m in deepseek_models)
        assert any(m.id == "deepseek-v4-flash" for m in deepseek_models)

    def test_get_model_specific(self):
        models = create_default_models()
        gpt4o = models.get_model("openai", "gpt-4o")
        assert gpt4o is not None
        assert gpt4o.id == "gpt-4o"
        assert gpt4o.provider == "openai"
        assert gpt4o.api == "openai-responses"

        ds_chat = models.get_model("deepseek", "deepseek-chat")
        assert ds_chat is not None
        assert ds_chat.api == "openai-completions"

    def test_get_model_missing(self):
        models = create_default_models()
        assert models.get_model("openai", "nonexistent") is None
        assert models.get_model("unknown", "gpt-4o") is None

    def test_add_remove_provider(self):
        models = Models()
        assert len(models.get_providers()) == 0

        p = openai_provider()
        models.add_provider(p)
        assert len(models.get_providers()) == 1
        assert models.get_provider("openai") is not None

        models.remove_provider("openai")
        assert len(models.get_providers()) == 0

    def test_unknown_provider_stream_raises(self):
        import asyncio

        models = Models()
        fake_model = Model(
            id="test",
            provider="nonexistent",
            api="openai-completions",
            name="Test",
        )
        ctx = Context(messages=[{"role": "user", "content": "hi"}])

        with pytest.raises(ValueError, match="Unknown provider"):
            asyncio.run(models.stream(fake_model, ctx))


class TestAuth:
    """Auth resolution."""

    def test_env_api_key_resolve_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        auth = env_api_key_auth("Test", ["TEST_API_KEY"])

        resolved = auth.resolve()
        assert resolved is not None
        assert resolved.api_key == "sk-test-123"
        assert resolved.source == "TEST_API_KEY"

    def test_env_api_key_not_set(self, monkeypatch):
        monkeypatch.delenv("NO_SUCH_VAR", raising=False)
        auth = env_api_key_auth("Test", ["NO_SUCH_VAR"])

        resolved = auth.resolve()
        assert resolved is None

    def test_in_memory_credential_store(self):
        store = InMemoryCredentialStore()
        cred = ApiKeyCredential(key="sk-memory")

        asyncio_sync(store.write("test", cred))
        result = asyncio_sync(store.read("test"))
        assert result is not None
        assert result.key == "sk-memory"

        asyncio_sync(store.delete("test"))
        assert asyncio_sync(store.read("test")) is None

    def test_multi_env_fallback(self, monkeypatch):
        """When the first env var is unset, fall back to the second."""
        monkeypatch.delenv("VAR_ONE", raising=False)
        monkeypatch.setenv("VAR_TWO", "sk-backup-456")

        auth = env_api_key_auth("Multi", ["VAR_ONE", "VAR_TWO"])
        resolved = auth.resolve()
        assert resolved is not None
        assert resolved.api_key == "sk-backup-456"
        assert resolved.source == "VAR_TWO"

    def test_credential_priority_over_env(self, monkeypatch):
        """Stored credential takes priority over environment variable."""
        monkeypatch.setenv("MY_API_KEY", "sk-from-env")

        auth = env_api_key_auth("Priority", ["MY_API_KEY"])
        cred = ApiKeyCredential(key="sk-from-credential")

        resolved = auth.resolve(credential=cred)
        assert resolved is not None
        assert resolved.api_key == "sk-from-credential"
        assert resolved.source == "stored credential"

    def test_resolve_no_credential_no_env(self, monkeypatch):
        """When neither credential nor env is available, resolve returns None."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        auth = env_api_key_auth("None", ["MISSING_KEY"])
        assert auth.resolve() is None


class TestResolveApiKey:
    """resolve_api_key() — Provider 使用的 API Key 解析。"""

    @pytest.mark.asyncio
    async def test_from_store(self):
        store = InMemoryCredentialStore()
        await store.write("deepseek", ApiKeyCredential(key="sk-stored"))
        auth = env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])

        api_key = await resolve_api_key(auth, store, "deepseek")
        assert api_key == "sk-stored"

    @pytest.mark.asyncio
    async def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        store = InMemoryCredentialStore()  # empty store
        auth = env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])

        api_key = await resolve_api_key(auth, store, "deepseek")
        assert api_key == "sk-from-env"

    @pytest.mark.asyncio
    async def test_store_preferred_over_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        store = InMemoryCredentialStore()
        await store.write("deepseek", ApiKeyCredential(key="sk-stored"))
        auth = env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])

        api_key = await resolve_api_key(auth, store, "deepseek")
        assert api_key == "sk-stored"

    @pytest.mark.asyncio
    async def test_missing_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
        store = InMemoryCredentialStore()  # empty store
        auth = env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"])

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            await resolve_api_key(auth, store, "deepseek")

    @pytest.mark.asyncio
    async def test_missing_raises_mentions_all_env_vars(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
        store = InMemoryCredentialStore()
        auth = env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"])

        with pytest.raises(ValueError) as excinfo:
            await resolve_api_key(auth, store, "deepseek")
        assert "DEEPSEEK_API_KEY" in str(excinfo.value)
        assert "DEEPSEEK_KEY" in str(excinfo.value)


class TestModelsSetApiKey:
    """Models.set_api_key() credential management."""

    @pytest.mark.asyncio
    async def test_set_and_read_api_key(self):
        models = create_default_models()
        await models.set_api_key("openai", "sk-test-openai")

        # Verify the key was stored in the shared credential store
        provider = models.get_provider("openai")
        assert provider is not None
        cred = await provider._credential_store.read("openai")
        assert cred is not None
        assert cred.key == "sk-test-openai"

    @pytest.mark.asyncio
    async def test_overwrite_api_key(self):
        models = create_default_models()
        await models.set_api_key("openai", "sk-first")
        await models.set_api_key("openai", "sk-second")

        provider = models.get_provider("openai")
        cred = await provider._credential_store.read("openai")
        assert cred.key == "sk-second"

    @pytest.mark.asyncio
    async def test_set_api_key_unknown_provider(self):
        """Setting API key for an unknown provider is allowed (writes to central store)."""
        models = Models()
        # set_api_key writes to the central credential store — provider validation
        # happens at stream() time, not at set_api_key() time.
        await models.set_api_key("nonexistent", "sk-xxx")
        # Should not raise; the key is stored in the central store


class TestModelsComplete:
    """Models.complete() — facade-level non-streaming call."""

    @pytest.mark.asyncio
    async def test_complete_returns_assistant_message(self):
        from unittest.mock import AsyncMock, patch
        from pi_ai._event_stream import AssistantMessageEventStream
        from pi_ai._types import AssistantMessage

        models = create_default_models()
        model = models.get_model("deepseek", "deepseek-chat")
        assert model is not None

        expected_msg = AssistantMessage(
            role="assistant",
            content=[{"type": "text", "text": "ok"}],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2},
            stopReason="stop",
            errorMessage=None,
            timestamp=0,
        )

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "message": expected_msg})

        provider = models.get_provider("deepseek")
        with patch.object(type(provider), "stream", new=AsyncMock(return_value=fake_stream)):
            ctx = Context(messages=[{"role": "user", "content": "hi"}])
            result = await models.complete(model, ctx)

        assert result["role"] == "assistant"
        assert result["stopReason"] == "stop"


def asyncio_sync(coro):
    """Helper to run async code in sync tests."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — create a new loop in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)
