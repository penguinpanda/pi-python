"""Prompt 缓存浪费统计（对齐 TS core/cache-stats.ts）。"""

from __future__ import annotations

from typing import Any

from pi_agent.compaction_utils import CACHE_TTL_MS, estimate_cache_state

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
    paid_cost = float(cost.get("input") or 0) + float(cost.get("cache_write") or 0)
    return paid_cost / paid_tokens if paid_tokens > 0 else 0.0


def _read_per_token(message: dict, price_source) -> float:
    usage = message.get("usage") or {}
    cost = usage.get("cost") or {}
    cache_read = _usage_field(usage, "cache_read")
    if cache_read > 0:
        return float(cost.get("cache_read") or 0) / cache_read
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


def detect_recent_cache_miss(messages: list[dict], price_source: Any = None) -> dict | None:
    """检测最近两轮请求间是否存在值得提示的单次缓存未命中。

    对齐 TS interactive-mode：missedTokens >= 20_000 或 missedCost >= 0.1 时
    返回 miss 详情，否则返回 None。低于 NOISE_FLOOR_TOKENS 的未命中忽略。
    """
    requests: list[dict] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        prompt_tokens = (
            _usage_field(usage, "input")
            + _usage_field(usage, "cache_read")
            + _usage_field(usage, "cache_write")
        )
        if prompt_tokens > 0:
            requests.append(message)
    if len(requests) < 2:
        return None

    prev, last = requests[-2], requests[-1]
    prev_usage = prev.get("usage") or {}
    last_usage = last.get("usage") or {}
    prev_prompt = (
        _usage_field(prev_usage, "input")
        + _usage_field(prev_usage, "cache_read")
        + _usage_field(prev_usage, "cache_write")
    )
    last_prompt = (
        _usage_field(last_usage, "input")
        + _usage_field(last_usage, "cache_read")
        + _usage_field(last_usage, "cache_write")
    )
    missed_tokens = min(prev_prompt, last_prompt) - _usage_field(last_usage, "cache_read")
    if missed_tokens <= NOISE_FLOOR_TOKENS:
        return None
    missed_cost = missed_tokens * max(
        0.0,
        _paid_per_token(last_usage) - _read_per_token(last, price_source),
    )
    if missed_tokens >= 20_000 or missed_cost >= 0.1:
        return {"missedTokens": missed_tokens, "missedCost": missed_cost, "missCount": 1}
    return None


__all__ = [
    "CACHE_TTL_MS",
    "NOISE_FLOOR_TOKENS",
    "compute_cache_waste",
    "detect_recent_cache_miss",
    "estimate_cache_state",
]
