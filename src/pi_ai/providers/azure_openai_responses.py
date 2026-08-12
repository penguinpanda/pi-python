"""Azure OpenAI Responses provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model, ModelCost

AZURE_OPENAI_RESPONSES_MODELS: list[Model] = [
    Model(
        id="gpt-5-chat-latest",
        provider="azure-openai-responses",
        api="azure-openai-responses",
        name="GPT-5 Chat Latest",
        input=["text", "image"],
        output=["text"],
        cost=ModelCost(input=1.25, output=10.0, cache_read=0.125),
        max_tokens=16384,
        context_window=128000,
    ),
    Model(
        id="gpt-5.6-luna",
        provider="azure-openai-responses",
        api="azure-openai-responses",
        name="GPT-5.6 Luna",
        input=["text", "image"],
        output=["text"],
        cost=ModelCost(input=0.2, output=1.2, cache_read=0.02, cache_write=0.25),
        max_tokens=128000,
        context_window=1050000,
        reasoning=True,
    ),
]


def azure_openai_responses_provider() -> Provider:
    return create_provider(
        id="azure-openai-responses",
        name="Azure OpenAI",
        auth=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"]),
        models=AZURE_OPENAI_RESPONSES_MODELS,
        base_url="",
        api_kind="azure-openai-responses",
    )


__all__ = [
    "AZURE_OPENAI_RESPONSES_MODELS",
    "azure_openai_responses_provider",
]
