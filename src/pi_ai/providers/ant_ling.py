"""Ant Ling provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.models.generated import load_generated_models
from pi_ai.provider import Provider, create_provider


def ant_ling_provider() -> Provider:
    return create_provider(
        id="ant-ling",
        name="Ant Ling",
        auth=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"]),
        models=load_generated_models().get("ant-ling", []),
        base_url="https://api.ant-ling.com/v1",
        api_kind="completions",
    )


__all__ = ["ant_ling_provider"]
