"""Token 费用计算（对齐 TS packages/ai/src/models.ts calculateCost）。"""

from __future__ import annotations

from ..types.message import Cost, Usage
from ..types.model import Model, ModelCostRates


def calculate_cost(model: Model, usage: Usage) -> Cost:
    """按模型单价与本次请求 token 用量计算费用（$），原地更新 usage.cost。

    单价为 $/百万 token；分档按 input + cacheRead + cacheWrite 命中
    inputTokensAbove 的最高档（严格大于阈值）；Anthropic cacheWrite1h
    按输入单价 2 倍计，其余 cacheWrite 按 cacheWrite 单价计。
    """
    input_tokens = usage["input"] + usage["cache_read"] + usage["cache_write"]
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    long_write = usage.get("cache_write_1h") or 0
    short_write = usage["cache_write"] - long_write
    cost = usage["cost"]
    cost["input"] = rates.input / 1_000_000 * usage["input"]
    cost["output"] = rates.output / 1_000_000 * usage["output"]
    cost["cache_read"] = rates.cache_read / 1_000_000 * usage["cache_read"]
    cost["cache_write"] = (
        rates.cache_write * short_write + rates.input * 2 * long_write
    ) / 1_000_000
    cost["total"] = cost["input"] + cost["output"] + cost["cache_read"] + cost["cache_write"]
    return cost


__all__ = ["calculate_cost"]
