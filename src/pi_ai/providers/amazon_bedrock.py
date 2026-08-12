"""AWS Bedrock provider（bearer token 认证）。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model

BEDROCK_MODELS: list[Model] = [
    Model(
        id="anthropic.claude-sonnet-4-20250514",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Claude Sonnet 4 on Bedrock",
        input=["text", "image"],
        output=["text"],
        max_tokens=32000,
        context_window=200000,
        reasoning=True,
    ),
    Model(
        id="amazon.nova-pro-v1:0",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Amazon Nova Pro",
        input=["text", "image"],
        output=["text"],
        max_tokens=32768,
        context_window=300000,
    ),
    Model(
        id="meta.llama3-1-405b-instruct-v1:0",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Llama 3.1 405B on Bedrock",
        input=["text"],
        output=["text"],
        max_tokens=32768,
        context_window=131072,
    ),
]


def amazon_bedrock_provider() -> Provider:
    return create_provider(
        id="amazon-bedrock",
        name="AWS Bedrock",
        auth=env_api_key_auth("AWS Bedrock bearer token", ["AWS_BEARER_TOKEN_BEDROCK"]),
        models=BEDROCK_MODELS,
        base_url="",
        api_kind="bedrock-converse-stream",
    )


__all__ = ["BEDROCK_MODELS", "amazon_bedrock_provider"]
