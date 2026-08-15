"""OpenRouter provider。"""

from __future__ import annotations

from pi_ai.auth import EnvApiKeyAuth
from pi_ai.auth.oauth.openrouter import open_router_oauth
from pi_ai.provider import Provider, create_provider


class _OpenRouterAuth:
    oauth = open_router_oauth
    display_name = "OpenRouter API key"
    env_vars = ["OPENROUTER_API_KEY"]

    def resolve(self, credential=None):  # type: ignore[no-untyped-def]
        return EnvApiKeyAuth(self.display_name, self.env_vars).resolve(credential)


def openrouter_provider() -> Provider:
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        auth=_OpenRouterAuth(),  # type: ignore[arg-type]
        # 模型由 create_default_models() 统一合并生成目录。
        models=[],
        base_url="https://openrouter.ai/api/v1",
        api_kind="completions",
    )


__all__ = ["openrouter_provider"]
