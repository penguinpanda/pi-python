"""Prompt 缓存浪费统计（对齐 TS core/cache-stats.ts）。"""

from __future__ import annotations

from typing import Any

# Prompt-cache TTL：超过该空闲时长的缓存未命中值得提示。
CACHE_TTL_MS = 5 * 60 * 1000
# 低于该 token 数的单轮未命中视为断点粒度噪声，不计数。
NOISE_FLOOR_TOKENS = 1024


def _usage_field(usage: dict | None, *keys: str) -> int:
    if not usage:
        return 0
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _as_previous_request(message: dict, reported_cache: bool) -> dict | None:
    usage = message.get("usage") or {}
    prompt_tokens = (
        _usage_field(usage, "input")
        + _usage_field(usage, "cache_read")
        + _usage_field(usage, "cache_write")
    )
    if prompt_tokens <= 0:
        return None
    return {
        "promptTokens": prompt_tokens,
        "modelKey": f"{message.get('provider', '?')}/{message.get('model', '?')}",
        "timestamp": _usage_field(message, "timestamp") or 0,
        "reportedCache": reported_cache
        or _usage_field(usage, "cache_read") + _usage_field(usage, "cache_write") > 0,
    }


def _paid_per_token(usage: dict) -> float:
    cost = usage.get("cost") or {}
    paid_tokens = _usage_field(usage, "input") + _usage_field(usage, "cache_write")
    paid_cost = float(cost.get("input") or 0) + float(cost.get("cacheWrite") or 0)
    return paid_cost / paid_tokens if paid_tokens > 0 else 0.0


def _read_per_token(message: dict, price_source) -> float:
    usage = message.get("usage") or {}
    cost = usage.get("cost") or {}
    cache_read = _usage_field(usage, "cache_read")
    if cache_read > 0:
        return float(cost.get("cacheRead") or 0) / cache_read
    model = None
    if price_source is not None and hasattr(price_source, "get_model"):
        try:
            model = price_source.get_model(message.get("provider", ""), message.get("model", ""))
        except Exception:
            model = None
    if model is not None:
        model_cost = getattr(model, "cost", None)
        cache_read_price = getattr(model_cost, "cache_read", None)
        if isinstance(cache_read_price, (int, float)):
            return float(cache_read_price) / 1_000_000
    return 0.0


def compute_cache_waste(messages: list[dict], price_source: Any = None) -> dict:
    """统计会话中的缓存浪费（对齐 TS computeCacheWaste）。

    返回 {missedTokens, missedCost, missCount}。missedCost 在定价未知时为 0。
    """
    totals = {"missedTokens": 0, "missedCost": 0.0, "missCount": 0}
    prev: dict | None = None

    for message in messages:
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        prompt_tokens = (
            _usage_field(usage, "input")
            + _usage_field(usage, "cache_read")
            + _usage_field(usage, "cache_write")
        )
        cache_tokens = _usage_field(usage, "cache_read") + _usage_field(usage, "cache_write")
        reported_cache = prev.get("reportedCache", False) if prev else False

        if prev is not None and prompt_tokens > 0 and (cache_tokens > 0 or reported_cache):
            missed_tokens = min(prev["promptTokens"], prompt_tokens) - _usage_field(
                usage, "cache_read"
            )
            if missed_tokens > NOISE_FLOOR_TOKENS:
                totals["missedTokens"] += missed_tokens
                totals["missCount"] += 1
                totals["missedCost"] += missed_tokens * max(
                    0.0,
                    _paid_per_token(usage) - _read_per_token(message, price_source),
                )

        next_prev = _as_previous_request(message, reported_cache)
        if next_prev is not None:
            prev = next_prev
    return totals


__all__ = ["CACHE_TTL_MS", "NOISE_FLOOR_TOKENS", "compute_cache_waste"]
