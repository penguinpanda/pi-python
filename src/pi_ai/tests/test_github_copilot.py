"""GitHub Copilot provider 测试。"""

from __future__ import annotations

from pi_ai import create_default_models
from pi_ai.providers.github_copilot import github_copilot_provider


def test_github_copilot_provider_registered() -> None:
    models = create_default_models()
    assert models.get_provider("github-copilot") is not None


def test_github_copilot_models_use_multi_api() -> None:
    provider = github_copilot_provider()
    apis = {model.api for model in provider.get_models()}
    assert "openai-completions" in apis
    assert "openai-responses" in apis


def test_github_copilot_has_oauth() -> None:
    provider = github_copilot_provider()
    assert getattr(provider.auth, "oauth", None) is not None
