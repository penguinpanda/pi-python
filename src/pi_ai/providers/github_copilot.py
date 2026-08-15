"""GitHub Copilot provider（OAuth + 多 API 模型分发）。"""

from __future__ import annotations

from typing import Any

from pi_ai.auth import EnvApiKeyAuth
from pi_ai.auth.oauth.github_copilot import github_copilot_oauth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model


def _filter_copilot_models(models: list[Model], credential: Any) -> list[Model]:
    """OAuth 登录后按 availableModelIds 过滤（对齐 TS filterModels）。"""
    if not isinstance(credential, dict) or credential.get("type") != "oauth":
        return models
    available = credential.get("available_model_ids")
    if not isinstance(available, list) or not all(isinstance(x, str) for x in available):
        return models
    allowed = set(available)
    return [model for model in models if model.id in allowed]


GITHUB_COPILOT_MODELS: list[Model] = [
    Model(
        id="gpt-4o",
        provider="github-copilot",
        api="openai-completions",
        name="GPT-4o",
        input=["text", "image"],
        output=["text"],
        max_tokens=16384,
        context_window=128000,
    ),
    Model(
        id="gpt-5-codex",
        provider="github-copilot",
        api="openai-responses",
        name="GPT-5 Codex",
        input=["text", "image"],
        output=["text"],
        max_tokens=16384,
        context_window=128000,
    ),
]


class _GithubCopilotAuth:
    oauth = github_copilot_oauth
    display_name = "GitHub Copilot token"
    env_vars = ["COPILOT_GITHUB_TOKEN"]

    def resolve(self, credential=None):  # type: ignore[no-untyped-def]
        return EnvApiKeyAuth(self.display_name, self.env_vars).resolve(credential)


def github_copilot_provider() -> Provider:
    return create_provider(
        id="github-copilot",
        name="GitHub Copilot",
        auth=_GithubCopilotAuth(),  # type: ignore[arg-type]
        models=GITHUB_COPILOT_MODELS,
        base_url="https://api.individual.githubcopilot.com",
        api_kind="completions",
        filter_models=_filter_copilot_models,
    )


__all__ = ["GITHUB_COPILOT_MODELS", "github_copilot_provider"]
