"""OAuth 流程包。"""

from ..types import OAuthAuth
from .device_code import poll_oauth_device_code_flow
from .pkce import generate_pkce
from . import github_copilot, openai_codex, openrouter


def builtin_oauth_providers() -> list[tuple[str, str, OAuthAuth]]:
    """内置 OAuth provider 列表：[(provider_id, display_name, OAuthAuth)]。"""
    return [
        (
            "openai-codex",
            openai_codex.openai_codex_oauth.name,
            openai_codex.openai_codex_oauth,
        ),
        (
            "github-copilot",
            github_copilot.github_copilot_oauth.name,
            github_copilot.github_copilot_oauth,
        ),
        (
            "openrouter",
            openrouter.open_router_oauth.name,
            openrouter.open_router_oauth,
        ),
    ]


__all__ = [
    "generate_pkce",
    "poll_oauth_device_code_flow",
    "builtin_oauth_providers",
]
