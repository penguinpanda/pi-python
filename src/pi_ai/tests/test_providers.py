"""
Unit tests for providers — openai_provider() / deepseek_provider() 工厂。

验证每个 Provider 工厂返回的配置：
    • id / name
    • API 类型（completions / responses）
    • Base URL
    • 认证方式（环境变量名）
    • 模型列表及能力元数据
"""

from pi_ai._types import Model, ModelCost
from pi_ai.auth import EnvApiKeyAuth
from pi_ai import create_default_models
from pi_ai.providers import (
    OLLAMA_MODELS,
    OPENAI_MODELS,
    QWEN_MODELS,
    QWEN_TOKEN_PLAN_BASE_URL,
    QWEN_TOKEN_PLAN_CN_BASE_URL,
    deepseek_provider,
    ollama_provider,
    openai_provider,
    qwen_provider,
    qwen_token_plan_cn_provider,
    qwen_token_plan_provider,
)
from pi_ai.providers.ollama import _QWEN3_THINKING_LEVEL_MAP, _merge_ollama_models

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
        assert _model_ids(models) == [
            "gpt-5-chat-latest",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ]

    def test_model_metadata(self):
        provider = openai_provider()
        by_id = {m.id: m for m in provider.get_models()}

        gpt5_chat = by_id["gpt-5-chat-latest"]
        assert gpt5_chat.api == "openai-responses"
        assert "image" in gpt5_chat.input
        assert gpt5_chat.reasoning is False
        assert gpt5_chat.context_window == 128000

        luna = by_id["gpt-5.6-luna"]
        assert luna.reasoning is True
        assert "image" in luna.input
        assert luna.context_window == 272000
        assert luna.cost.tiers[0].input_tokens_above == 272000


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
        # 生成目录模型由 create_default_models() 统一合并（工厂不再重复加载）。
        provider = create_default_models().get_provider("deepseek")
        models = provider.get_models()
        assert _model_ids(models) == [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ]

    def test_model_metadata(self):
        provider = create_default_models().get_provider("deepseek")
        by_id = {m.id: m for m in provider.get_models()}

        v4_flash = by_id["deepseek-v4-flash"]
        assert v4_flash.api == "openai-responses"
        assert v4_flash.reasoning is True
        assert v4_flash.max_tokens == 384000
        assert v4_flash.context_window == 1000000
        assert v4_flash.cost.input == 0.14
        assert v4_flash.cost.output == 0.28
        assert v4_flash.cost.cache_read == 0.0028
        assert v4_flash.cost.cache_write == 0.0
        assert v4_flash.thinking_level_map == {
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "max": "max",
        }
        assert v4_flash.compat == {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
            "supportsExplicitPromptCacheMode": False,
            "supportsLongCacheRetention": False,
            "supportsWebSearch": True,
        }

        v4_pro = by_id["deepseek-v4-pro"]
        assert v4_pro.api == "openai-responses"
        assert v4_pro.reasoning is True
        assert v4_pro.max_tokens == 384000
        assert v4_pro.context_window == 1000000
        assert v4_pro.cost.input == 0.435
        assert v4_pro.cost.output == 0.87
        assert v4_pro.cost.cache_read == 0.003625
        assert v4_pro.cost.cache_write == 0.0
        assert v4_pro.thinking_level_map == {
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "max": "max",
        }
        assert v4_pro.compat == {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
            "supportsExplicitPromptCacheMode": False,
            "supportsLongCacheRetention": False,
            "supportsWebSearch": True,
        }


class TestQwenProvider:
    """qwen_provider() 工厂。"""

    def test_factory_config(self):
        provider = qwen_provider()
        assert provider.id == "qwen"
        assert provider.name == "Qwen"
        assert provider._api_kind == "completions"
        assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_auth_uses_env_var(self):
        provider = qwen_provider()
        auth = provider.auth
        assert isinstance(auth, EnvApiKeyAuth)
        assert auth.display_name == "Qwen (DashScope) API key"
        assert auth.env_vars == ["DASHSCOPE_API_KEY", "QWEN_API_KEY"]

    def test_model_list(self):
        provider = qwen_provider()
        models = provider.get_models()
        assert _model_ids(models) == [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen3-235b-a22b",
            "qwen3-30b-a3b",
            "qwen3-vl-flash",
            "qwen-vl-plus",
            "qwen-vl-max",
        ]

    def test_model_metadata(self):
        provider = qwen_provider()
        by_id = {m.id: m for m in provider.get_models()}

        plus = by_id["qwen-plus"]
        assert plus.api == "openai-completions"
        assert plus.reasoning is False
        assert plus.context_window == 131072

        thinking = by_id["qwen3-235b-a22b"]
        assert thinking.reasoning is True

        vision = by_id["qwen-vl-plus"]
        assert "image" in vision.input

        vl_flash = by_id["qwen3-vl-flash"]
        assert "image" in vl_flash.input


class TestModelConstants:
    """OPENAI_MODELS / OLLAMA_MODELS 常量及 DeepSeek 生成目录。"""

    def test_openai_models_constant(self):
        assert _model_ids(OPENAI_MODELS) == [
            "gpt-5-chat-latest",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ]

    def test_deepseek_models_from_generated_catalog(self):
        # 生成目录模型由 create_default_models() 统一合并（工厂不再重复加载）。
        provider = create_default_models().get_provider("deepseek")
        assert _model_ids(provider.get_models()) == [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
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
            "qwen3-14b-q5km-coding",
            "qwen38-q3km-16k",
        ]

    def test_qwen_models_constant(self):
        assert _model_ids(QWEN_MODELS) == [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen3-235b-a22b",
            "qwen3-30b-a3b",
            "qwen3-vl-flash",
            "qwen-vl-plus",
            "qwen-vl-max",
        ]

    def test_all_models_have_provider_and_api(self):
        deepseek_models = create_default_models().get_provider("deepseek").get_models()
        for model in OPENAI_MODELS + OLLAMA_MODELS + deepseek_models:
            assert model.provider in ("openai", "deepseek", "ollama")
            assert model.api in ("openai-completions", "openai-responses")


class TestModelMetadata:
    """Model 能力元数据：reasoning 字段 + input 模态列表。"""

    def test_reasoning_and_image_input(self):
        m = Model(
            id="x",
            provider="p",
            api="a",
            reasoning=True,
            input=["text", "image"],
        )
        assert m.reasoning is True
        assert "image" in m.input

    def test_defaults(self):
        m = Model(id="x", provider="p", api="a")
        assert m.reasoning is False
        assert "image" not in m.input

    def test_text_only(self):
        m = Model(
            id="x",
            provider="p",
            api="a",
            reasoning=True,
            input=["text"],
        )
        assert m.reasoning is True
        assert "image" not in m.input


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
            "qwen3-14b-q5km-coding",
            "qwen38-q3km-16k",
        ]

    def test_model_metadata(self):
        provider = ollama_provider()
        by_id = {m.id: m for m in provider.get_models()}

        vision = by_id["llama3.2-vision:latest"]
        assert vision.api == "openai-completions"
        assert "image" in vision.input
        assert vision.reasoning is False

        r1 = by_id["deepseek-r1:14b"]
        assert r1.reasoning is True

        qwen3 = by_id["qwen3:30b"]
        assert qwen3.reasoning is True
        assert qwen3.thinking_level_map == _QWEN3_THINKING_LEVEL_MAP
        assert qwen3.thinking_level_map["off"] == "none"

        qwen38 = by_id["qwen38-q3km-16k"]
        assert qwen38.reasoning is True
        assert qwen38.thinking_level_map == _QWEN3_THINKING_LEVEL_MAP
        assert qwen38.thinking_level_map["off"] == "none"

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
        assert known.reasoning is True
        # 未知模型合成默认元数据
        unknown = models[1]
        assert unknown.provider == "ollama"
        assert unknown.api == "openai-completions"
        assert unknown.cost == ModelCost()

    def test_merge_order_follows_api_tags(self):
        models = _merge_ollama_models(["deepseek-r1:14b", "qwen3:30b"])
        assert [m.id for m in models] == ["deepseek-r1:14b", "qwen3:30b"]
        # 复用静态对象元数据
        assert models[1].reasoning is True

    def test_provider_accepts_discovered_models(self):
        discovered = _merge_ollama_models(["qwen3:30b", "brand-new:latest"])
        provider = ollama_provider(models=discovered)
        assert [m.id for m in provider.get_models()] == [
            "qwen3:30b",
            "brand-new:latest",
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
            ollama_mod.httpx,
            "AsyncClient",
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

        monkeypatch.setattr(ollama_mod.httpx, "AsyncClient", lambda **kw: _FakeClient())

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

        monkeypatch.setattr(ollama_mod.httpx, "AsyncClient", lambda **kw: _FakeClient())

        assert await ollama_mod.discover_ollama_models() is None


class TestQwenTokenPlanProviders:
    """qwen_token_plan_provider() / qwen_token_plan_cn_provider() 工厂。

    对齐 TS test/qwen-token-plan-models.test.ts 的覆盖范围。
    """

    TEXT_MODELS = [
        "MiniMax-M2.5",
        "deepseek-v3.2",
        "deepseek-v4-flash",
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "glm-5",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2.5",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "qwen3.6-flash",
        "qwen3.6-plus",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-max",
    ]

    # 图像/视频生成模型不支持工具调用，生成目录应排除。
    EXCLUDED_MODELS = [
        "qwen-image-2.0",
        "qwen-image-2.0-pro",
        "wan2.7-image",
        "wan2.7-image-pro",
        "happyhorse-1.1-t2v",
        # 已退役的 preview id。
        "qwen3.8-max-preview",
    ]

    def test_factory_config(self):
        provider = qwen_token_plan_provider()
        assert provider.id == "qwen-token-plan"
        assert provider.name == "Qwen Token Plan"
        assert provider._api_kind == "completions"
        assert (
            provider.base_url
            == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )

        cn_provider = qwen_token_plan_cn_provider()
        assert cn_provider.id == "qwen-token-plan-cn"
        assert cn_provider.name == "Qwen Token Plan CN"
        assert cn_provider._api_kind == "completions"
        assert (
            cn_provider.base_url
            == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )

    def test_auth_uses_env_var(self):
        auth = qwen_token_plan_provider().auth
        assert isinstance(auth, EnvApiKeyAuth)
        assert auth.display_name == "Qwen Token Plan API key"
        assert auth.env_vars == ["QWEN_TOKEN_PLAN_API_KEY"]

        cn_auth = qwen_token_plan_cn_provider().auth
        assert isinstance(cn_auth, EnvApiKeyAuth)
        assert cn_auth.display_name == "Qwen Token Plan CN API key"
        assert cn_auth.env_vars == ["QWEN_TOKEN_PLAN_CN_API_KEY"]

    def test_model_catalog(self):
        models = create_default_models()
        for provider in (
            models.get_provider("qwen-token-plan"),
            models.get_provider("qwen-token-plan-cn"),
        ):
            ids = _model_ids(provider.get_models())
            assert ids == sorted(ids), provider.id
            for expected in self.TEXT_MODELS:
                assert expected in ids, f"{provider.id} missing {expected}"
            for excluded in self.EXCLUDED_MODELS:
                assert excluded not in ids, f"{provider.id} includes {excluded}"

    def test_both_regions_share_catalog_with_distinct_endpoints(self):
        models = create_default_models()
        intl = models.get_provider("qwen-token-plan")
        cn = models.get_provider("qwen-token-plan-cn")
        assert _model_ids(intl.get_models()) == _model_ids(cn.get_models())
        intl_by_id = {m.id: m for m in intl.get_models()}
        cn_by_id = {m.id: m for m in cn.get_models()}
        for model_id in self.TEXT_MODELS:
            assert intl_by_id[model_id].base_url == QWEN_TOKEN_PLAN_BASE_URL
            assert cn_by_id[model_id].base_url == QWEN_TOKEN_PLAN_CN_BASE_URL

    def test_model_metadata(self):
        by_id = {
            m.id: m for m in create_default_models().get_provider("qwen-token-plan").get_models()
        }

        qwen38 = by_id["qwen3.8-max"]
        assert qwen38.provider == "qwen-token-plan"
        assert qwen38.api == "openai-completions"
        assert qwen38.reasoning is True
        assert "image" in qwen38.input
        assert qwen38.context_window == 1000000
        assert qwen38.max_tokens == 131072
        # Token Plan 为订阅制套餐，按 token 计费为 0。
        assert qwen38.cost.input == 0.0
        assert qwen38.cost.output == 0.0
        assert qwen38.compat == {
            "thinkingFormat": "qwen",
            "supportsDeveloperRole": False,
            "supportsStore": False,
            "supportsReasoningEffort": True,
        }
        assert qwen38.thinking_level_map == {
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": None,
            "xhigh": "xhigh",
            "max": None,
        }

        # 不支持 reasoning_effort 的模型：关闭 effort 且无思考级别映射。
        kimi = by_id["kimi-k2.7-code"]
        assert kimi.reasoning is True
        assert kimi.compat is not None
        assert kimi.compat.get("supportsReasoningEffort") is False
        assert kimi.compat.get("thinkingFormat") == "qwen"
        assert kimi.thinking_level_map is None

        # 其余支持 effort 的推理模型映射 high/max（qwen3.8-max 除外）。
        v4_flash = by_id["deepseek-v4-flash"]
        assert v4_flash.thinking_level_map == {
            "minimal": None,
            "low": None,
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": "max",
        }

    def test_custom_models_override(self):
        custom = Model(
            id="custom-model",
            provider="qwen-token-plan",
            api="openai-completions",
            name="Custom",
        )
        provider = qwen_token_plan_provider(models=[custom])
        assert _model_ids(provider.get_models()) == ["custom-model"]
