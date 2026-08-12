"""OpenAI 兼容 completions provider 批量注册（动态 /models 发现）。"""

from __future__ import annotations

from typing import Any

import httpx

from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, RefreshModelsContext, create_provider
from pi_ai.types import Model

_AsyncClient = httpx.AsyncClient


def _credential_key(credential: Any) -> str | None:
    if credential is None:
        return None
    if isinstance(credential, dict):
        return credential.get("key")
    return getattr(credential, "key", None)


async def _fetch_openai_models(
    provider_id: str,
    base_url: str,
    env_key: str,
    context: RefreshModelsContext,
) -> list[Model]:
    if not context.allow_network:
        return []
    api_key = _credential_key(context.credential)
    if not api_key:
        return []
    async with _AsyncClient() as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
    rows = response.json().get("data") or []
    return [
        Model(
            id=str(row.get("id", "")),
            provider=provider_id,
            api="openai-completions",
            name=str(row.get("id", "")),
            input=["text"],
            output=["text"],
        )
        for row in rows
        if row.get("id")
    ]


def _provider(
    provider_id: str,
    name: str,
    base_url: str,
    env_key: str,
) -> Provider:
    async def fetch(context: RefreshModelsContext) -> list[Model]:
        return await _fetch_openai_models(provider_id, base_url, env_key, context)

    return create_provider(
        id=provider_id,
        name=name,
        auth=env_api_key_auth(name, [env_key]),
        models=[],
        base_url=base_url,
        api_kind="completions",
        fetch_models=fetch,
    )


def groq_provider() -> Provider:
    return _provider("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY")


def together_provider() -> Provider:
    return _provider(
        "together",
        "Together AI",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
    )


def cerebras_provider() -> Provider:
    return _provider(
        "cerebras",
        "Cerebras",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
    )


def fireworks_provider() -> Provider:
    return _provider(
        "fireworks",
        "Fireworks AI",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
    )


def nvidia_provider() -> Provider:
    return _provider(
        "nvidia",
        "NVIDIA NIM",
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
    )


def huggingface_provider() -> Provider:
    return _provider(
        "huggingface",
        "Hugging Face",
        "https://router.huggingface.co/v1",
        "HF_TOKEN",
    )


def baseten_provider() -> Provider:
    return _provider(
        "baseten",
        "Baseten",
        "https://model-apis.baseten.co/v1",
        "BASETEN_API_KEY",
    )


def moonshotai_provider() -> Provider:
    return _provider(
        "moonshotai",
        "Moonshot AI",
        "https://api.moonshot.cn/v1",
        "MOONSHOT_API_KEY",
    )


def xiaomi_provider() -> Provider:
    return _provider(
        "xiaomi",
        "Xiaomi MiMo",
        "https://api.xiaomimimo.com/v1",
        "XIAOMI_API_KEY",
    )


def zai_provider() -> Provider:
    return _provider(
        "zai",
        "Z.ai",
        "https://api.lingyiwanwu.com/v1",
        "ZAI_API_KEY",
    )


def xai_provider() -> Provider:
    from pi_ai.auth.oauth.xai import xai_oauth

    class _XaiAuth:
        oauth = xai_oauth
        display_name = "xAI API key"
        env_vars = ["XAI_API_KEY"]

        def resolve(self, credential=None):  # type: ignore[no-untyped-def]
            return env_api_key_auth(self.display_name, self.env_vars).resolve(credential)

    return create_provider(
        id="xai",
        name="xAI",
        auth=_XaiAuth(),  # type: ignore[arg-type]
        models=[],
        base_url="https://api.x.ai/v1",
        api_kind="completions",
        fetch_models=lambda context: _fetch_openai_models(
            "xai", "https://api.x.ai/v1", "XAI_API_KEY", context
        ),
    )


def moonshotai_cn_provider() -> Provider:
    return _provider(
        "moonshotai-cn",
        "Moonshot AI CN",
        "https://api.moonshot.cn/v1",
        "MOONSHOT_API_KEY",
    )


def zai_coding_cn_provider() -> Provider:
    return _provider(
        "zai-coding-cn",
        "Z.AI Coding CN",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "ZAI_CODING_CN_API_KEY",
    )


def opencode_provider() -> Provider:
    return _provider(
        "opencode",
        "OpenCode Zen",
        "https://opencode.ai/api/v1",
        "OPENCODE_API_KEY",
    )


def opencode_go_provider() -> Provider:
    return _provider(
        "opencode-go",
        "OpenCode Go",
        "https://opencode.ai/api/v1",
        "OPENCODE_API_KEY",
    )


def xiaomi_token_plan_ams_provider() -> Provider:
    return _provider(
        "xiaomi-token-plan-ams",
        "Xiaomi Token Plan AMS",
        "https://token-plan-ams.xiaomimimo.com/v1",
        "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    )


def xiaomi_token_plan_cn_provider() -> Provider:
    return _provider(
        "xiaomi-token-plan-cn",
        "Xiaomi Token Plan CN",
        "https://token-plan-cn.xiaomimimo.com/v1",
        "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    )


def xiaomi_token_plan_sgp_provider() -> Provider:
    return _provider(
        "xiaomi-token-plan-sgp",
        "Xiaomi Token Plan SGP",
        "https://token-plan-sgp.xiaomimimo.com/v1",
        "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    )


__all__ = [
    "groq_provider",
    "together_provider",
    "cerebras_provider",
    "fireworks_provider",
    "nvidia_provider",
    "huggingface_provider",
    "baseten_provider",
    "moonshotai_provider",
    "xiaomi_provider",
    "zai_provider",
    "xai_provider",
    "moonshotai_cn_provider",
    "zai_coding_cn_provider",
    "opencode_provider",
    "opencode_go_provider",
    "xiaomi_token_plan_ams_provider",
    "xiaomi_token_plan_cn_provider",
    "xiaomi_token_plan_sgp_provider",
]
