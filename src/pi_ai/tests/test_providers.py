"""
Unit tests for providers — openai_provider() / deepseek_provider() 工厂。

验证每个 Provider 工厂返回的配置：
    • id / name
    • API 类型（completions / responses）
    • Base URL
    • 认证方式（环境变量名）
    • 模型列表及能力元数据
"""

from pi_ai._types import Model, ModelCapabilities, ModelCost
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.providers import (
    DEEPSEEK_MODELS,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    deepseek_provider,
    ollama_provider,
    openai_provider,
)
from pi_ai.providers.ollama import _merge_ollama_models, discover_ollama_models

import httpx


def _model_ids(models: list[Model]) -> list[str]:
    return [m.id for m in models]


class TestOpenAIProvider:
    """openai_provider() 工厂。"""

    def test_factory_config(self):
        provider = openai_provider()
        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert provider._api_kind == "responses"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_auth_uses_env_var(self):
        provider = openai_provider()
        auth = provider.auth
        assert isinstance(auth, EnvApiKeyAuth)
        assert auth.display_name == "OpenAI API key"
        assert auth.env_vars == ["OPENAI_API_KEY"]

    def test_model_list(self):
        provider = openai_provider()
        models = provider.get_models()
        assert _model_ids(models) == ["gpt-4o", "gpt-4o-mini", "o4-mini"]

    def test_model_capabilities(self):
        provider = openai_provider()
        by_id = {m.id: m for m in provider.get_models()}

        gpt4o = by_id["gpt-4o"]
        assert gpt4o.api == "openai-responses"
        assert "image" in gpt4o.input
        assert gpt4o.capabilities.images is True
        assert gpt4o.capabilities.tools is True
        assert gpt4o.capabilities.reasoning is False

        o4_mini = by_id["o4-mini"]
        assert o4_mini.capabilities.reasoning is True
        assert o4_mini.capabilities.tools is True
        assert o4_mini.capabilities.images is False


class TestDeepSeekProvider:
    """deepseek_provider() 工厂。"""

    def test_factory_config(self):
        provider = deepseek_provider()
        assert provider.id == "deepseek"
        assert provider.name == "DeepSeek"
        assert provider._api_kind == "completions"
        assert provider.base_url == "https://api.deepseek.com"

    def test_auth_uses_env_var(self):
        provider = deepseek_provider()
        auth = provider.auth
        assert isinstance(auth, EnvApiKeyAuth)
        assert auth.display_name == "DeepSeek API key"
        assert auth.env_vars == ["DEEPSEEK_API_KEY"]

    def test_model_list(self):
        provider = deepseek_provider()
        models = provider.get_models()
        assert _model_ids(models) == ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"]

    def test_model_capabilities(self):
        provider = deepseek_provider()
        by_id = {m.id: m for m in provider.get_models()}

        chat = by_id["deepseek-chat"]
        assert chat.api == "openai-completions"
        assert chat.capabilities.tools is True
        assert chat.capabilities.reasoning is False

        reasoner = by_id["deepseek-reasoner"]
        assert reasoner.api == "openai-completions"
        assert reasoner.capabilities.reasoning is True
        assert reasoner.capabilities.tools is False

        v4_flash = by_id["deepseek-v4-flash"]
        assert v4_flash.api == "openai-completions"
        assert v4_flash.capabilities.reasoning is True
        assert v4_flash.capabilities.tools is True
        assert v4_flash.max_tokens == 384000


class TestModelConstants:
    """OPENAI_MODELS / DEEPSEEK_MODELS / OLLAMA_MODELS 常量。"""

    def test_openai_models_constant(self):
        assert _model_ids(OPENAI_MODELS) == ["gpt-4o", "gpt-4o-mini", "o4-mini"]

    def test_deepseek_models_constant(self):
        assert _model_ids(DEEPSEEK_MODELS) == [
            "deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash",
        ]

    def test_ollama_models_constant(self):
        assert _model_ids(OLLAMA_MODELS) == [
            "qwen3:30b",
            "qwen3:30b-a3b",
            "richardyoung/qwen3-14b-abliterated:Q5_K_M",
            "gpt-oss:20b",
            "llama3.2-vision:latest",
            "qwen2.5:7b-instruct-q8_0",
            "deepseek-r1:14b",
        ]

    def test_all_models_have_provider_and_api(self):
        for model in OPENAI_MODELS + DEEPSEEK_MODELS + OLLAMA_MODELS:
            assert model.provider in ("openai", "deepseek", "ollama")
            assert model.api in ("openai-completions", "openai-responses")


class TestModelCapabilities:
    """Model.capabilities() 能力标签。"""

    def test_capabilities_all(self):
        m = Model(
            id="x", provider="p", api="a",
            capabilities=ModelCapabilities(reasoning=True, tools=True, images=True),
        )
        assert m.capabilities.reasoning is True
        assert m.capabilities.tools is True
        assert m.capabilities.images is True

    def test_capabilities_none(self):
        m = Model(id="x", provider="p", api="a")
        assert m.capabilities.reasoning is False
        assert m.capabilities.tools is False
        assert m.capabilities.images is False

    def test_capabilities_partial(self):
        m = Model(
            id="x", provider="p", api="a",
            capabilities=ModelCapabilities(reasoning=True, images=True),
        )
        assert m.capabilities.reasoning is True
        assert m.capabilities.images is True
        assert m.capabilities.tools is False


class TestOllamaProvider:
    """ollama_provider() 工厂。"""

    def test_factory_config(self):
        provider = ollama_provider()
        assert provider.id == "ollama"
        assert provider.name == "Ollama"
        assert provider._api_kind == "completions"
        assert provider.base_url == "http://127.0.0.1:11434/v1"

    def test_no_auth_required(self):
        """本地服务不需要 API Key，auth 为 None。"""
        provider = ollama_provider()
        assert provider.auth is None

    def test_model_list(self):
        provider = ollama_provider()
        models = provider.get_models()
        assert _model_ids(models) == [
            "qwen3:30b",
            "qwen3:30b-a3b",
            "richardyoung/qwen3-14b-abliterated:Q5_K_M",
            "gpt-oss:20b",
            "llama3.2-vision:latest",
            "qwen2.5:7b-instruct-q8_0",
            "deepseek-r1:14b",
        ]

    def test_model_capabilities(self):
        provider = ollama_provider()
        by_id = {m.id: m for m in provider.get_models()}

        vision = by_id["llama3.2-vision:latest"]
        assert vision.api == "openai-completions"
        assert "image" in vision.input
        assert vision.capabilities.images is True
        assert vision.capabilities.reasoning is False

        r1 = by_id["deepseek-r1:14b"]
        assert r1.capabilities.reasoning is True
        assert r1.capabilities.tools is False

        qwen3 = by_id["qwen3:30b"]
        assert qwen3.capabilities.reasoning is True
        assert qwen3.capabilities.tools is True

    def test_all_models_are_local_and_free(self):
        for model in OLLAMA_MODELS:
            assert model.provider == "ollama"
            assert model.api == "openai-completions"
            assert model.cost == ModelCost()


class TestOllamaDiscovery:
    """Ollama 运行时动态发现。"""

    def test_merge_known_and_unknown(self):
        models = _merge_ollama_models(["qwen3:30b", "brand-new:latest"])
        assert [m.id for m in models] == ["qwen3:30b", "brand-new:latest"]
        # 已知模型保留静态元数据
        known = models[0]
        assert known.capabilities.reasoning is True
        assert known.capabilities.tools is True
        # 未知模型合成默认元数据
        unknown = models[1]
        assert unknown.provider == "ollama"
        assert unknown.api == "openai-completions"
        assert unknown.capabilities.tools is True
        assert unknown.cost == ModelCost()

    def test_merge_order_follows_api_tags(self):
        models = _merge_ollama_models(["deepseek-r1:14b", "qwen3:30b"])
        assert [m.id for m in models] == ["deepseek-r1:14b", "qwen3:30b"]
        # 复用静态对象元数据
        assert models[1].capabilities.reasoning is True

    def test_provider_accepts_discovered_models(self):
        discovered = _merge_ollama_models(["qwen3:30b", "brand-new:latest"])
        provider = ollama_provider(models=discovered)
        assert [m.id for m in provider.get_models()] == [
            "qwen3:30b", "brand-new:latest",
        ]

    async def test_discover_success(self, monkeypatch):
        from pi_ai.providers import ollama as ollama_mod

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"models": [{"name": "qwen3:30b"}, {"name": "new:latest"}]}

        class _FakeClient:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                assert url.endswith("/api/tags")
                return self._response

        monkeypatch.setattr(
            ollama_mod.httpx, "AsyncClient",
            lambda **kw: _FakeClient(_FakeResponse()),
        )

        models = await ollama_mod.discover_ollama_models()
        assert models is not None
        assert [m.id for m in models] == ["qwen3:30b", "new:latest"]

    async def test_discover_connection_error_returns_none(self, monkeypatch):
        from pi_ai.providers import ollama as ollama_mod

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(
            ollama_mod.httpx, "AsyncClient", lambda **kw: _FakeClient()
        )

        assert await ollama_mod.discover_ollama_models() is None

    async def test_discover_non_200_returns_none(self, monkeypatch):
        from pi_ai.providers import ollama as ollama_mod

        class _FakeResponse:
            status_code = 500

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                return _FakeResponse()

        monkeypatch.setattr(
            ollama_mod.httpx, "AsyncClient", lambda **kw: _FakeClient()
        )

        assert await ollama_mod.discover_ollama_models() is None
