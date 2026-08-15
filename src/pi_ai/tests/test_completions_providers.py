"""openai-completions provider 配置测试。"""

from __future__ import annotations

import pytest

from pi_ai import Models
from pi_ai.auth.context import AuthContext
from pi_ai import create_default_models
from pi_ai.providers.ant_ling import ant_ling_provider
from pi_ai.providers.openrouter import openrouter_provider


class _EnvContext(AuthContext):
    def __init__(self, env: dict[str, str]) -> None:
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


def test_openrouter_provider_config() -> None:
    provider = openrouter_provider()
    assert provider.id == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert hasattr(provider.auth, "oauth")
    # 模型由 create_default_models() 统一合并生成目录。
    models = create_default_models().get_provider("openrouter").get_models()
    assert models
    assert {model.api for model in models} == {"openai-completions"}


def test_ant_ling_provider_config() -> None:
    provider = ant_ling_provider()
    assert provider.id == "ant-ling"
    assert provider.base_url == "https://api.ant-ling.com/v1"
    assert provider.auth is not None and provider.auth.env_vars == ["ANT_LING_API_KEY"]
    models = create_default_models().get_provider("ant-ling").get_models()
    assert len(models) == 3


@pytest.mark.asyncio
async def test_openrouter_env_api_key() -> None:
    models = Models(auth_context=_EnvContext({"OPENROUTER_API_KEY": "sk-or"}))
    models.add_provider(openrouter_provider())
    result = await models.get_auth("openrouter")
    assert result is not None
    assert result.auth["api_key"] == "sk-or"


@pytest.mark.asyncio
async def test_dynamic_models_inherit_static_metadata_and_detect_compat(monkeypatch) -> None:
    """动态 /models 发现应继承静态目录元数据并重新检测 provider compat。"""
    import httpx

    from pi_ai.providers.openai_completions_providers import _fetch_openai_models, zai_provider

    provider = zai_provider()
    assert provider.get_models()
    static_model = next(m for m in provider.get_models() if m.id == "glm-4.6")
    assert static_model.context_window > 0
    assert static_model.reasoning is True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "glm-4.6"}, {"id": "custom-new-model"}]})

    monkeypatch.setattr(
        "pi_ai.providers.openai_completions_providers._AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    context = type(
        "Ctx",
        (),
        {
            "allow_network": True,
            "credential": {"type": "api_key", "key": "zai-key"},
        },
    )()
    discovered = await _fetch_openai_models(
        "zai",
        "https://api.z.ai/api/coding/paas/v4",
        "ZAI_API_KEY",
        context,
        static_models={m.id: m for m in provider.get_models()},
    )
    by_id = {m.id: m for m in discovered}
    assert by_id["glm-4.6"].context_window == static_model.context_window
    assert by_id["glm-4.6"].thinking_level_map == static_model.thinking_level_map
    # 直连 z.ai 不得继承 OpenRouter 的 thinkingFormat=openrouter。
    assert by_id["glm-4.6"].compat["thinkingFormat"] == "zai"
    assert by_id["custom-new-model"].compat["thinkingFormat"] == "zai"
