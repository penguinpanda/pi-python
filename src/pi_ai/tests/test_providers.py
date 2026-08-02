"""
Unit tests for providers — openai_provider() / deepseek_provider() 工厂。

验证每个 Provider 工厂返回的配置：
    • id / name
    • API 类型（completions / responses）
    • Base URL
    • 认证方式（环境变量名）
    • 模型列表及能力元数据
"""

from pi_ai._types import Model
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.providers import (
    DEEPSEEK_MODELS,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    deepseek_provider,
    ollama_provider,
    openai_provider,
)


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
        assert gpt4o.supportsImages is True
        assert gpt4o.supportsToolCalling is True
        assert gpt4o.thinking is False

        o4_mini = by_id["o4-mini"]
        assert o4_mini.thinking is True
        assert o4_mini.supportsToolCalling is True
        assert o4_mini.supportsImages is False


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
        assert chat.supportsToolCalling is True
        assert chat.thinking is False

        reasoner = by_id["deepseek-reasoner"]
        assert reasoner.api == "openai-completions"
        assert reasoner.thinking is True
        assert reasoner.supportsToolCalling is False

        v4_flash = by_id["deepseek-v4-flash"]
        assert v4_flash.api == "openai-completions"
        assert v4_flash.thinking is True
        assert v4_flash.supportsToolCalling is True
        assert v4_flash.maxTokens == 384000


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
        assert vision.supportsImages is True
        assert vision.thinking is False

        r1 = by_id["deepseek-r1:14b"]
        assert r1.thinking is True
        assert r1.supportsToolCalling is False

        qwen3 = by_id["qwen3:30b"]
        assert qwen3.thinking is True
        assert qwen3.supportsToolCalling is True

    def test_all_models_are_local_and_free(self):
        for model in OLLAMA_MODELS:
            assert model.provider == "ollama"
            assert model.api == "openai-completions"
            assert model.cost == {}
