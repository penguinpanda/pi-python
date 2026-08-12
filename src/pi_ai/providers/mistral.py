"""Mistral provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model, ModelCost

MISTRAL_MODELS: list[Model] = [
    Model(
        id="mistral-medium-3.5",
        provider="mistral",
        api="mistral-conversations",
        name="Mistral Medium 3.5",
        input=["text", "image"],
        output=["text"],
        cost=ModelCost(input=1.5, output=7.5),
        max_tokens=262144,
        context_window=262144,
        reasoning=True,
    )
]


def mistral_provider() -> Provider:
    return create_provider(
        id="mistral",
        name="Mistral",
        auth=env_api_key_auth("Mistral", ["MISTRAL_API_KEY"]),
        models=MISTRAL_MODELS,
        base_url="https://api.mistral.ai/v1",
        api_kind="mistral-conversations",
    )


__all__ = ["MISTRAL_MODELS", "mistral_provider"]
