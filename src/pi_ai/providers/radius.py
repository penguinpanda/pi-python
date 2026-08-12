"""Radius 网关 provider（pi-messages API + 动态 catalog）。

对齐 TS providers/radius.ts + radius-config.ts：模型目录从网关
`GET /v1/config` 拉取（auth: Bearer <api key>），缓存于 ModelsStore。
"""

from __future__ import annotations

from typing import Any

import httpx

from pi_ai.provider import EnvApiKeyAuth, Provider, RefreshModelsContext, create_provider
from pi_ai.types import Model, ModelCost

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"

# 测试可替换的 client 工厂（间接层避免 monkeypatch httpx.AsyncClient 全局递归）。
_client_factory = httpx.AsyncClient


def normalize_radius_gateway_url(value: str) -> str:
    """补 https:// 并去尾斜杠（对齐 TS normalizeRadiusGatewayUrl）。"""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value.rstrip("/")


def _sanitize_gateway_config(config: Any) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    base_url = config.get("baseUrl")
    models = config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for raw in models:
        if not isinstance(raw, dict):
            continue
        if not (
            isinstance(raw.get("id"), str)
            and isinstance(raw.get("name"), str)
            and isinstance(raw.get("reasoning"), bool)
            and isinstance(raw.get("input"), list)
            and isinstance(raw.get("cost"), dict)
            and isinstance(raw.get("contextWindow"), int)
            and isinstance(raw.get("maxTokens"), int)
        ):
            continue
        cleaned.append(
            {
                k: v
                for k, v in raw.items()
                if k
                in (
                    "id",
                    "name",
                    "reasoning",
                    "thinkingLevelMap",
                    "input",
                    "cost",
                    "contextWindow",
                    "maxTokens",
                )
            }
        )
    return {"baseUrl": base_url, "models": cleaned}


def get_radius_models_from_config(provider_id: str, config: dict[str, Any]) -> list[Model]:
    """网关配置 → Model 列表（对齐 TS getRadiusModelsFromConfig）。"""
    base_url = str(config["baseUrl"])
    models: list[Model] = []
    for raw in config["models"]:
        cost = _to_model_cost(raw.get("cost"))
        models.append(
            Model(
                id=raw["id"],
                name=raw["name"],
                reasoning=bool(raw.get("reasoning", False)),
                input=[str(item) for item in raw.get("input", ["text"])],  # type: ignore[misc]
                cost=cost,
                context_window=int(raw.get("contextWindow", 0)),
                max_tokens=int(raw.get("maxTokens", 0)),
                thinking_level_map=raw.get("thinkingLevelMap"),
                api="pi-messages",
                provider=provider_id,
                base_url=base_url,
            )
        )
    return models


def _to_model_cost(raw: Any) -> ModelCost:
    if not isinstance(raw, dict):
        return ModelCost()
    rates: dict[str, float] = {}
    for key, default in (
        ("input", 0.0),
        ("output", 0.0),
        ("cacheRead", 0.0),
        ("cacheWrite", 0.0),
    ):
        value = raw.get(key)
        rates[key] = float(value) if isinstance(value, (int, float)) else default
    tiers = []
    for tier in raw.get("tiers") or []:
        if isinstance(tier, dict):
            tiers.append(
                {
                    "input_tokens_above": int(tier.get("inputTokensAbove", 0)),
                    "input": float(tier.get("input", 0)),
                    "output": float(tier.get("output", 0)),
                    "cache_read": float(tier.get("cacheRead", 0)),
                    "cache_write": float(tier.get("cacheWrite", 0)),
                }
            )
    return ModelCost(
        input=rates["input"],
        output=rates["output"],
        cache_read=rates["cacheRead"],
        cache_write=rates["cacheWrite"],
        tiers=tiers,  # type: ignore[arg-type]
    )


async def _fetch_radius_models(
    provider_id: str,
    gateway: str,
    context: RefreshModelsContext,
) -> list[Model]:
    if not context.allow_network:
        return []
    if context.signal is not None and context.signal.is_set():
        return []
    credential = context.credential
    api_key = None
    if isinstance(credential, dict):
        api_key = credential.get("key") or credential.get("access")
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    try:
        async with _client_factory(timeout=15.0) as client:
            response = await client.get(f"{gateway}/v1/config", headers=headers)
            response.raise_for_status()
            config = _sanitize_gateway_config(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Could not load Radius config from {gateway}: {exc}") from exc
    if config is None:
        raise RuntimeError(f"Invalid Radius config from {gateway}")
    if context.signal is not None and context.signal.is_set():
        return []
    return get_radius_models_from_config(provider_id, config)


def radius_provider(
    provider_id: str = "radius",
    name: str = "Radius",
    gateway: str = DEFAULT_RADIUS_GATEWAY,
) -> Provider:
    """Radius 网关 provider（对齐 TS radiusProvider）。"""
    normalized = normalize_radius_gateway_url(gateway)
    return create_provider(
        id=provider_id,
        name=name,
        auth=EnvApiKeyAuth(name, ["RADIUS_API_KEY"]),
        models=[],
        api_kind="pi-messages",
        fetch_models=lambda context: _fetch_radius_models(provider_id, normalized, context),
    )
