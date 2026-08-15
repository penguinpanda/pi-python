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
