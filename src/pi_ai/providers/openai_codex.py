"""OpenAI Codex provider。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from pi_ai.auth import EnvApiKeyAuth
from pi_ai.auth.oauth.openai_codex import openai_codex_oauth
from pi_ai.api.openai_codex_responses import (
    codex_cancel_deferred,
    codex_fetch_deferred,
)
from pi_ai.models.generated import load_generated_models
from pi_ai.provider import Provider, create_provider


class _OpenAICodexAuth:
    oauth = openai_codex_oauth
    display_name = "OpenAI Codex"
    env_vars = ["OPENAI_API_KEY"]

    def resolve(self, credential=None):  # type: ignore[no-untyped-def]
        return EnvApiKeyAuth(self.display_name, self.env_vars).resolve(credential)


def openai_codex_provider() -> Provider:
    models = []
    for model in load_generated_models().get("openai-codex", []):
        compat = dict(model.compat or {})
        compat["supportsToolSearch"] = True
        models.append(replace(model, compat=cast(Any, compat)))
    return create_provider(
        id="openai-codex",
        name="OpenAI Codex",
        auth=_OpenAICodexAuth(),  # type: ignore[arg-type]
        models=models,
        base_url="https://chatgpt.com/backend-api",
        api_kind="openai-codex-responses",
        deferred_fn=codex_fetch_deferred,
        cancel_deferred_fn=codex_cancel_deferred,
    )


__all__ = ["openai_codex_provider"]
