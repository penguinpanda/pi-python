"""Google Vertex AI provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model

GOOGLE_VERTEX_MODELS: list[Model] = [
    Model(
        id="gemini-2.5-pro",
        provider="google-vertex",
        api="google-vertex",
        name="Vertex Gemini 2.5 Pro",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
    Model(
        id="gemini-2.5-flash",
        provider="google-vertex",
        api="google-vertex",
        name="Vertex Gemini 2.5 Flash",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
]


def google_vertex_provider() -> Provider:
    return create_provider(
        id="google-vertex",
        name="Google Vertex",
        auth=env_api_key_auth("Google Vertex", ["GOOGLE_OAUTH_ACCESS_TOKEN"]),
        models=GOOGLE_VERTEX_MODELS,
        base_url="",
        api_kind="google-vertex",
    )


__all__ = ["GOOGLE_VERTEX_MODELS", "google_vertex_provider"]
