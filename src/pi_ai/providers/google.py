"""Google Gemini provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model

GOOGLE_MODELS: list[Model] = [
    Model(
        id="gemini-2.5-pro",
        provider="google",
        api="google-generative-ai",
        name="Gemini 2.5 Pro",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
    Model(
        id="gemini-2.5-flash",
        provider="google",
        api="google-generative-ai",
        name="Gemini 2.5 Flash",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
]


def google_provider() -> Provider:
    return create_provider(
        id="google",
        name="Google",
        auth=env_api_key_auth("Gemini API key", ["GEMINI_API_KEY"]),
        models=GOOGLE_MODELS,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_kind="google-generative-ai",
    )


__all__ = ["GOOGLE_MODELS", "google_provider"]
