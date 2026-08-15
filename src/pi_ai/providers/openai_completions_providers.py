"""OpenAI 兼容 completions provider 批量注册（动态 /models 发现）。

动态目录作为 overlay；静态能力元数据来自 Python 已生成的 TS 目录
（OpenRouter / Vercel AI Gateway），按 provider 前缀反推同款直连模型。
"""

from __future__ import annotations

import json

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx

from pi_ai.api.compat_runtime import detect_completions_compat
from pi_ai.auth import env_api_key_auth
from pi_ai.provider import Provider, RefreshModelsContext, create_provider
from pi_ai.types import Model

_AsyncClient = httpx.AsyncClient

# OpenRouter / Vercel 目录中的上游 provider 前缀 → Python provider id。
_STATIC_CATALOG_PREFIXES: dict[str, tuple[str, ...]] = {
    "moonshotai": ("moonshotai",),
    "moonshotai-cn": ("moonshotai",),
    "zai": ("z-ai", "zai"),
    "zai-coding-cn": ("z-ai", "zai"),
    "xai": ("x-ai", "xai"),
    "nvidia": ("nvidia",),
    "xiaomi": ("xiaomi",),
    "mistral": ("mistralai", "mistral"),
    "groq": ("groq",),
    "together": ("together", "together_ai"),
    "cerebras": ("cerebras",),
    "fireworks": ("fireworks", "fireworks_ai"),
    "huggingface": ("huggingface",),
    "baseten": ("baseten",),
    "opencode": ("opencode",),
}


def _generated_provider_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "generated" / "providers"


@lru_cache(maxsize=1)
def _static_provider_models() -> dict[str, dict[str, Model]]:
    """从已生成的 TS 目录反推直连 provider 的静态模型元数据。

    返回 {provider_id: {model_id: Model}}。字段（cost/contextWindow/
    maxTokens/reasoning/thinkingLevelMap/compat）原样继承；api 统一改为
    openai-completions，provider/id 改为直连 provider。
    """
    result: dict[str, dict[str, Model]] = {}
    for filename in ("openrouter.json", "vercel-ai-gateway.json"):
        path = _generated_provider_dir() / filename
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for model_id, data in raw.items():
            if not isinstance(data, dict) or "/" not in model_id:
                continue
            prefix, _, suffix = model_id.partition("/")
            if not suffix:
                continue
            for provider_id, prefixes in _STATIC_CATALOG_PREFIXES.items():
                if prefix not in prefixes:
                    continue
                static = result.setdefault(provider_id, {})
                if suffix in static:
                    continue
                cost_data = data.get("cost") or {}
                from pi_ai.types import ModelCost, ModelCostTier

                raw_compat = dict(data["compat"]) if isinstance(data.get("compat"), dict) else None
                if raw_compat is not None:
                    # 这些是 OpenRouter/网关路由键，直接 provider 不应继承；
                    # 请求时由 detect_completions_compat 重新检测。
                    for gateway_key in (
                        "thinkingFormat",
                        "supportsDeveloperRole",
                        "cacheControlFormat",
                        "sessionAffinityFormat",
                        "sendSessionAffinityHeaders",
                        "openRouterRouting",
                        "vercelGatewayRouting",
                    ):
                        raw_compat.pop(gateway_key, None)

                tiers = [
                    ModelCostTier(
                        input=float(tier.get("input", 0.0) or 0.0),
                        output=float(tier.get("output", 0.0) or 0.0),
                        cache_read=float(tier.get("cacheRead", 0.0) or 0.0),
                        cache_write=float(tier.get("cacheWrite", 0.0) or 0.0),
                        input_tokens_above=int(tier.get("inputTokensAbove", 0) or 0),
                    )
                    for tier in cost_data.get("tiers") or []
                    if isinstance(tier, dict)
                ]

                static[suffix] = Model(
                    id=suffix,
                    provider=provider_id,
                    api="openai-completions",
                    name=str(data.get("name") or suffix),
                    input=list(data.get("input") or ["text"]),
                    output=list(data.get("output") or ["text"]),
                    cost=ModelCost(
                        input=float(cost_data.get("input", 0.0) or 0.0),
                        output=float(cost_data.get("output", 0.0) or 0.0),
                        cache_read=float(cost_data.get("cacheRead", 0.0) or 0.0),
                        cache_write=float(cost_data.get("cacheWrite", 0.0) or 0.0),
                        tiers=tiers,
                    ),
                    max_tokens=int(data.get("max_tokens", 4096) or 4096),
                    context_window=int(data.get("context_window", 0) or 0),
                    compat=cast(Any, raw_compat),
                    thinking_level_map=(
                        dict(data["thinking_level_map"])
                        if isinstance(data.get("thinking_level_map"), dict)
                        else None
                    ),
                    reasoning=bool(data.get("reasoning", False)),
                )
                break
    return result


def _static_models_for(provider_id: str) -> dict[str, Model]:
    return dict(_static_provider_models().get(provider_id, {}))


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
    *,
    static_models: dict[str, Model] | None = None,
) -> list[Model]:
    if not context.allow_network:
        return []
    api_key = _credential_key(context.credential)
    if not api_key:
        return []
    async with _AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
    rows = response.json().get("data") or []
    static = static_models or {}
    discovered: list[Model] = []
    for row in rows:
        model_id = row.get("id")
        if not model_id:
            continue
        model_id = str(model_id)
        base = static.get(model_id)
        if base is not None:
            model = replace(base)
        else:
            model = Model(
                id=model_id,
                provider=provider_id,
                api="openai-completions",
                name=str(row.get("name") or model_id),
                input=["text"],
                output=["text"],
            )
        # 远端行偶尔会带能力字段（部分兼容端点）；优先于合成默认值。
        if isinstance(row.get("context_window"), int) and row["context_window"] > 0:
            model.context_window = row["context_window"]
        if isinstance(row.get("max_tokens"), int) and row["max_tokens"] > 0:
            model.max_tokens = row["max_tokens"]
        if isinstance(row.get("reasoning"), bool):
            model.reasoning = row["reasoning"]
        # provider 级 detectCompat：即使没有静态目录，请求参数也能对齐 TS。
        detected_compat = detect_completions_compat(model)
        model.compat = cast(
            Any,
            {
                **detected_compat,
                **(model.compat or {}),
            },
        )
        discovered.append(model)
    return discovered


def _provider(
    provider_id: str,
    name: str,
    base_url: str,
    env_key: str,
    *,
    models_base_url: str | None = None,
) -> Provider:
    fetch_base_url = models_base_url or base_url
    static_models = _static_models_for(provider_id)

    async def fetch(context: RefreshModelsContext) -> list[Model]:
        return await _fetch_openai_models(
            provider_id,
            fetch_base_url,
            env_key,
            context,
            static_models=static_models,
        )

    return create_provider(
        id=provider_id,
        name=name,
        auth=env_api_key_auth(name, [env_key]),
        models=list(static_models.values()),
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
        "https://api.together.ai/v1",
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
        "https://api.fireworks.ai/inference",
        "FIREWORKS_API_KEY",
        models_base_url="https://api.fireworks.ai/inference/v1",
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
        "https://inference.baseten.co/v1",
        "BASETEN_API_KEY",
    )


def moonshotai_provider() -> Provider:
    return _provider(
        "moonshotai",
        "Moonshot AI",
        "https://api.moonshot.ai/v1",
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
        "https://api.z.ai/api/coding/paas/v4",
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

    static_models = _static_models_for("xai")
    return create_provider(
        id="xai",
        name="xAI",
        auth=_XaiAuth(),  # type: ignore[arg-type]
        models=list(static_models.values()),
        base_url="https://api.x.ai/v1",
        api_kind="completions",
        fetch_models=lambda context: _fetch_openai_models(
            "xai",
            "https://api.x.ai/v1",
            "XAI_API_KEY",
            context,
            static_models=static_models,
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
