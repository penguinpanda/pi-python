"""Ant Ling provider。"""

from __future__ import annotations

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, create_provider


def ant_ling_provider() -> Provider:
    return create_provider(
        id="ant-ling",
        name="Ant Ling",
        auth=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"]),
        # 模型由 create_default_models() 统一合并生成目录。
        models=[],
        base_url="https://api.ant-ling.com/v1",
        api_kind="completions",
    )


__all__ = ["ant_ling_provider"]
