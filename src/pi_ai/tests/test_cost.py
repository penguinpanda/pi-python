"""calculate_cost 单元测试（对齐 TS packages/ai calculateCost）。"""

from __future__ import annotations

import pytest

from pi_ai import Model, ModelCost, ModelCostTier
from pi_ai.types.message import Usage
from pi_ai.utils.cost import calculate_cost


def _usage(
    *,
    input: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cache_write_1h: int | None = None,
) -> Usage:
    usage = Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input + output + cache_read + cache_write,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )
    if cache_write_1h is not None:
        usage["cache_write_1h"] = cache_write_1h
    return usage


def _model(cost: ModelCost) -> Model:
    return Model(id="m", provider="p", api="openai-completions", cost=cost)


def test_basic_rates():
    model = _model(ModelCost(input=0.14, output=0.28, cache_read=0.0028, cache_write=0.0))
    usage = _usage(input=1_000_000, output=500_000, cache_read=100_000)
    calculate_cost(model, usage)
    cost = usage["cost"]
    assert cost["input"] == pytest.approx(0.14)
    assert cost["output"] == pytest.approx(0.14)
    assert cost["cache_read"] == pytest.approx(0.00028)
    assert cost["total"] == pytest.approx(0.28028)


def test_tier_selection_uses_highest_matching_threshold():
    model = _model(
        ModelCost(
            input=0.1,
            output=0.2,
            cache_read=0.0,
            cache_write=0.0,
            tiers=[
                ModelCostTier(
                    input=0.05,
                    output=0.1,
                    cache_read=0.0,
                    cache_write=0.0,
                    input_tokens_above=1_000,
                ),
                ModelCostTier(
                    input=0.02,
                    output=0.04,
                    cache_read=0.0,
                    cache_write=0.0,
                    input_tokens_above=10_000,
                ),
            ],
        )
    )
    base_rate = 0.1 / 1_000_000
    u500 = _usage(input=500)
    calculate_cost(model, u500)
    assert u500["cost"]["input"] == pytest.approx(base_rate * 500)
    # TS 边界：inputTokens > inputTokensAbove 才命中，1000 不命中第一档。
    u1000 = _usage(input=1_000)
    calculate_cost(model, u1000)
    assert u1000["cost"]["input"] == pytest.approx(base_rate * 1_000)
    u5000 = _usage(input=5_000)
    calculate_cost(model, u5000)
    assert u5000["cost"]["input"] == pytest.approx(0.05 / 1_000_000 * 5_000)
    u20000 = _usage(input=20_000)
    calculate_cost(model, u20000)
    assert u20000["cost"]["input"] == pytest.approx(0.02 / 1_000_000 * 20_000)


def test_cache_write_1h_charged_at_double_input_rate():
    model = _model(ModelCost(input=1.0, output=0.0, cache_read=0.0, cache_write=0.1))
    usage = _usage(cache_write=1_000, cache_write_1h=1_000)
    calculate_cost(model, usage)
    cost = usage["cost"]
    assert cost["cache_write"] == pytest.approx(2.0 * 1_000 / 1_000_000)
    assert cost["total"] == pytest.approx(cost["cache_write"])


def test_short_and_long_cache_write_split():
    model = _model(ModelCost(input=1.0, output=0.0, cache_read=0.0, cache_write=0.1))
    usage = _usage(cache_write=2_000, cache_write_1h=500)
    calculate_cost(model, usage)
    cost = usage["cost"]
    assert cost["cache_write"] == pytest.approx((0.1 * 1_500 + 1.0 * 2 * 500) / 1_000_000)


def test_cache_write_1h_exceeds_cache_write_clamped():
    """异常 usage（cache_write_1h > cache_write）不得产生负费用。"""
    model = _model(ModelCost(input=1.0, output=0.0, cache_read=0.0, cache_write=0.1))
    usage = _usage(cache_write=100, cache_write_1h=1000)
    calculate_cost(model, usage)
    assert usage["cost"]["cache_write"] >= 0
