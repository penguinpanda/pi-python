"""ModelRuntime / ModelRegistry 单元测试（Faux Provider，零网络）。"""

from __future__ import annotations

import pytest

from pi_ai import Context, Models, UserMessage
from pi_ai.auth import ApiKeyCredential
from pi_ai.providers.faux import FAUX_MODEL, faux_assistant_message, faux_provider
from pi_ai.providers.openai import openai_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_config import ModelConfig
from pi_coding_agent.model_registry import ModelRegistry
from pi_coding_agent.model_runtime import ModelRuntime


async def _make_runtime(
    providers=None,
    config: ModelConfig | None = None,
    auth_store: AuthStorage | None = None,
) -> ModelRuntime:
    store = auth_store or AuthStorage.in_memory()
    models = Models(credentials=store)
    for provider in providers or [faux_provider().provider]:
        models.add_provider(provider)
    runtime = ModelRuntime(models, store, config=config)
    await runtime.get_available()
    return runtime


class TestNoAuthProvider:
    async def test_faux_is_configured(self):
        runtime = await _make_runtime()
        check = await runtime.check_auth("faux")
        assert check is not None
        assert check["type"] == "api_key"

    async def test_get_available_includes_faux(self):
        runtime = await _make_runtime()
        available = await runtime.get_available()
        assert any(model.id == "faux-1" for model in available)

    async def test_get_auth_no_auth_required(self):
        runtime = await _make_runtime()
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        resolution = await runtime.get_auth(model)
        assert resolution is not None
        assert resolution.source == "no auth required"


class TestStoredCredentials:
    async def test_stored_credential_configures_provider(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store = AuthStorage.in_memory()

        async def _set(_current):
            return ApiKeyCredential(key="sk-stored")

        await store.modify("openai", _set)
        runtime = await _make_runtime(providers=[openai_provider()], auth_store=store)

        assert runtime.has_configured_auth("openai")
        check = await runtime.check_auth("openai")
        assert check["source"] == "stored credential"

        resolution = await runtime.get_auth("openai")
        assert resolution is not None
        assert resolution.auth["api_key"] == "sk-stored"

    async def test_env_var_configures_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        runtime = await _make_runtime(providers=[openai_provider()])
        assert await runtime.check_auth("openai") is not None
        resolution = await runtime.get_auth("openai")
        assert resolution.auth["api_key"] == "sk-env"

    async def test_stored_preferred_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        store = AuthStorage.in_memory()

        async def _set(_current):
            return ApiKeyCredential(key="sk-stored")

        await store.modify("openai", _set)
        runtime = await _make_runtime(providers=[openai_provider()], auth_store=store)
        resolution = await runtime.get_auth("openai")
        assert resolution.auth["api_key"] == "sk-stored"

    async def test_unconfigured_provider_not_available(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runtime = await _make_runtime(providers=[openai_provider()])
        assert await runtime.check_auth("openai") is None
        available = await runtime.get_available()
        assert all(model.provider != "openai" for model in available)


class TestModelsJsonComposition:
    async def test_configured_api_key_and_base_url(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(
            """
            {
              "providers": {
                "openai": {
                  "baseUrl": "https://custom.api/v1",
                  "apiKey": "sk-config"
                }
              }
            }
            """,
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        runtime = await _make_runtime(providers=[openai_provider()], config=config)

        assert runtime.has_configured_auth("openai")
        resolution = await runtime.get_auth("openai")
        assert resolution.auth["api_key"] == "sk-config"

        provider = runtime.get_provider("openai")
        assert provider is not None
        assert provider.base_url == "https://custom.api/v1"

    async def test_model_override_applied(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(
            """
            {
              "providers": {
                "openai": {
                  "apiKey": "sk-x",
                  "modelOverrides": {
                    "gpt-5-chat-latest": { "reasoning": true, "maxTokens": 4096 }
                  }
                }
              }
            }
            """,
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        runtime = await _make_runtime(providers=[openai_provider()], config=config)
        model = runtime.get_model("openai", "gpt-5-chat-latest")
        assert model is not None
        assert model.reasoning is True
        assert model.max_tokens == 4096

    async def test_configured_headers_resolved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_PI_CUSTOM_HEADER", "hv")
        path = tmp_path / "models.json"
        path.write_text(
            """
            {
              "providers": {
                "faux": {
                  "apiKey": "sk-faux",
                  "headers": { "X-Custom": "$TEST_PI_CUSTOM_HEADER" },
                  "authHeader": true
                }
              }
            }
            """,
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        runtime = await _make_runtime(config=config)
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        resolution = await runtime.get_auth(model)
        assert resolution is not None
        assert resolution.auth["api_key"] == "sk-faux"
        headers = resolution.auth["headers"]
        assert headers["X-Custom"] == "hv"
        assert headers["Authorization"] == "Bearer sk-faux"

    async def test_composition_error_recorded(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(
            '{"providers": {"acme": {"models": [{"id": "m1"}]}}}',
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        runtime = await _make_runtime(config=config)
        error = runtime.get_error()
        assert error is not None
        assert 'Provider "acme"' in error


class TestExtensionProviders:
    async def test_register_provider_config(self):
        runtime = await _make_runtime()
        runtime.register_provider(
            "acme",
            {
                "api_key": "sk-acme",
                "base_url": "https://acme.api/v1",
                "models": [
                    {
                        "id": "acme-1",
                        "name": "Acme One",
                        "api": "openai-completions",
                        "reasoning": True,
                    }
                ],
            },
        )
        model = runtime.get_model("acme", "acme-1")
        assert model is not None
        assert model.base_url == "https://acme.api/v1"
        assert runtime.has_configured_auth("acme")

        resolution = await runtime.get_auth("acme")
        assert resolution.auth["api_key"] == "sk-acme"

    async def test_unregister_provider(self):
        runtime = await _make_runtime()
        runtime.register_provider(
            "acme",
            {
                "api_key": "sk-acme",
                "models": [{"id": "m1", "api": "openai-completions", "base_url": "https://x"}],
            },
        )
        assert runtime.get_provider("acme") is not None
        runtime.unregister_provider("acme")
        assert runtime.get_provider("acme") is None

    async def test_register_native_provider(self):
        runtime = await _make_runtime()
        core = faux_provider(models=[FAUX_MODEL])
        runtime.register_native_provider(core.provider)
        assert runtime.get_provider("faux") is not None


class TestRuntimeApiKey:
    async def test_set_and_remove_runtime_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runtime = await _make_runtime(providers=[openai_provider()])
        assert not runtime.has_configured_auth("openai")

        await runtime.set_runtime_api_key("openai", "sk-runtime")
        assert runtime.has_configured_auth("openai")
        resolution = await runtime.get_auth("openai")
        assert resolution.auth["api_key"] == "sk-runtime"

        await runtime.remove_runtime_api_key("openai")
        assert not runtime.has_configured_auth("openai")


class TestModelRegistry:
    async def test_facade(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store = AuthStorage.in_memory()

        async def _set(_current):
            return ApiKeyCredential(key="sk-stored")

        await store.modify("openai", _set)
        runtime = await _make_runtime(
            providers=[openai_provider(), faux_provider().provider],
            auth_store=store,
        )
        registry = ModelRegistry(runtime)

        available = registry.get_available()
        assert any(model.provider == "openai" for model in available)
        assert any(model.provider == "faux" for model in available)

        model = registry.find("openai", "gpt-5-chat-latest")
        assert model is not None
        assert registry.has_configured_auth(model)

        result = await registry.get_api_key_and_headers(model)
        assert result["ok"] is True
        assert result["api_key"] == "sk-stored"

    async def test_find_by_id_and_display_name(self):
        provider = faux_provider().provider
        runtime = await _make_runtime(providers=[provider])
        registry = ModelRegistry(runtime)
        assert registry.find_by_id("faux-1") is not None
        assert registry.find_by_id("missing") is None
        assert registry.get_provider_display_name("faux") == provider.name
        assert registry.get_provider_display_name("unknown") == "unknown"

    def test_register_by_name_requires_config(self):
        runtime = None  # 仅验证参数校验，不触碰 runtime
        registry = ModelRegistry(runtime)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="config is required"):
            registry.register_provider("openai")

    async def test_api_key_and_headers_with_faux_resolution_none(self):
        runtime = await _make_runtime(providers=[faux_provider().provider])
        registry = ModelRegistry(runtime)
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        result = await registry.get_api_key_and_headers(model)
        assert result["ok"] is True
        assert result["api_key"] is None


class TestModelRegistryAuthBranches:
    def _model(self):
        from pi_ai import Model

        return Model(id="m", provider="p", api="a")

    async def test_resolution_headers_filter_none(self):
        class _Runtime:
            async def get_auth(self, model):
                return type(
                    "Resolution",
                    (),
                    {"auth": {"api_key": "k", "headers": {"X": "v", "Y": None}}, "env": None},
                )()

            def get_compatibility_request_config(self, model):
                return {}

        result = await ModelRegistry(_Runtime()).get_api_key_and_headers(self._model())  # type: ignore[arg-type]
        assert result["ok"] is True
        assert result["api_key"] == "k"
        assert result["headers"] == {"X": "v"}

    async def test_resolution_none_with_auth_header_errors(self):
        class _Runtime:
            async def get_auth(self, model):
                return None

            def get_compatibility_request_config(self, model):
                return {"auth_header": "Authorization"}

        result = await ModelRegistry(_Runtime()).get_api_key_and_headers(self._model())  # type: ignore[arg-type]
        assert result["ok"] is False
        assert "No API key found" in result["error"]

    async def test_api_key_for_provider_swallows_errors(self):
        class _Runtime:
            async def get_auth(self, provider):
                raise RuntimeError("boom")

        assert await ModelRegistry(_Runtime()).get_api_key_for_provider("p") is None  # type: ignore[arg-type]


class TestRuntimeStream:
    async def test_stream_through_runtime(self):
        core = faux_provider()
        core.set_responses([faux_assistant_message("runtime stream ok")])
        runtime = await _make_runtime(providers=[core.provider])
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        context = Context(messages=[UserMessage(role="user", content="hi")])
        stream = await runtime.stream(model, context)
        message = await stream.result()
        text = "".join(
            block.get("text", "") for block in message["content"] if block.get("type") == "text"
        )
        assert text == "runtime stream ok"

    async def test_stream_with_configured_headers(self, tmp_path):
        core = faux_provider()
        core.set_responses([faux_assistant_message("with headers ok")])
        path = tmp_path / "models.json"
        path.write_text(
            """
            {
              "providers": {
                "faux": {
                  "apiKey": "sk-faux",
                  "headers": { "X-Custom": "hv" }
                }
              }
            }
            """,
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        runtime = await _make_runtime(providers=[core.provider], config=config)
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        context = Context(messages=[UserMessage(role="user", content="hi")])
        stream = await runtime.stream(model, context)
        message = await stream.result()
        text = "".join(
            block.get("text", "") for block in message["content"] if block.get("type") == "text"
        )
        assert text == "with headers ok"
