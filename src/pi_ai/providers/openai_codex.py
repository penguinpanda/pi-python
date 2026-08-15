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
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model


class _OpenAICodexAuth:
    oauth = openai_codex_oauth
    display_name = "OpenAI Codex"
    env_vars = ["OPENAI_API_KEY"]

    def resolve(self, credential=None):  # type: ignore[no-untyped-def]
        return EnvApiKeyAuth(self.display_name, self.env_vars).resolve(credential)


def apply_codex_compat(model: Model) -> Model:
    """OpenAI Codex 模型统一开启 tool search（生成目录中部分模型缺失该标志）。"""
    compat = dict(model.compat or {})
    compat["supportsToolSearch"] = True
    return replace(model, compat=cast(Any, compat))


def openai_codex_provider() -> Provider:
    return create_provider(
        id="openai-codex",
        name="OpenAI Codex",
        auth=_OpenAICodexAuth(),  # type: ignore[arg-type]
        # 模型由 create_default_models() 统一合并生成目录（合并时应用
        # apply_codex_compat 后处理，见 providers/all.py）。
        models=[],
        base_url="https://chatgpt.com/backend-api",
        api_kind="openai-codex-responses",
        deferred_fn=codex_fetch_deferred,
        cancel_deferred_fn=codex_cancel_deferred,
    )


__all__ = ["apply_codex_compat", "openai_codex_provider"]
