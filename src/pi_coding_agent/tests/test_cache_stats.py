"""Prompt 缓存浪费统计测试（对齐 TS cache-stats.ts）。"""

from __future__ import annotations

from pi_coding_agent.cache_stats import NOISE_FLOOR_TOKENS, compute_cache_waste


def _assistant(provider="faux", model="faux-1", usage=None):
    return {
        "role": "assistant",
        "provider": provider,
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "usage": usage or {},
    }


class TestComputeCacheWaste:
    def test_empty(self):
        assert compute_cache_waste([]) == {
            "missedTokens": 0,
            "missedCost": 0.0,
            "missCount": 0,
        }

    def test_first_turn_not_counted(self):
        messages = [
            _assistant(usage={"input": 20000, "cache_read": 0, "cache_write": 0}),
        ]
        totals = compute_cache_waste(messages)
        assert totals["missCount"] == 0
        assert totals["missedTokens"] == 0

    def test_miss_above_noise_floor_counted(self):
        messages = [
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
            _assistant(
                usage={
                    "input": 5000,
                    "cache_read": 15000,
                    "cache_write": 0,
                    "cost": {
                        "input": 0.01,
                        "cacheWrite": 0.0,
                        "cacheRead": 0.005,
                    },
                }
            ),
        ]
        totals = compute_cache_waste(messages)
        assert totals["missCount"] == 1
        assert totals["missedTokens"] == 5000
        assert totals["missedCost"] > 0

    def test_miss_below_noise_floor_ignored(self):
        messages = [
            _assistant(usage={"input": 5000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 4000, "cache_read": 4000, "cache_write": 0}),
        ]
        totals = compute_cache_waste(messages)
        assert totals["missCount"] == 0
        assert totals["missedTokens"] == 0

    def test_provider_without_cache_reporting_ignored(self):
        messages = [
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
        ]
        totals = compute_cache_waste(messages)
        assert totals["missCount"] == 0
        assert totals["missedTokens"] == 0

    def test_model_price_fallback(self):
        class _PriceSource:
            def get_model(self, provider, model_id):
                class _Cost:
                    cache_read = 0.002

                class _Model:
                    cost = _Cost()

                return _Model()

        messages = [
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 20000, "cache_read": 10000, "cache_write": 0}),
        ]
        totals = compute_cache_waste(messages, _PriceSource())
        assert totals["missCount"] == 1
        assert totals["missedTokens"] == 20000
        assert totals["missedCost"] == 0.0

    def test_non_assistant_messages_skipped(self):
        messages = [
            {"role": "user", "content": "hi"},
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
            {"role": "user", "content": "again"},
            _assistant(usage={"input": 20000, "cache_read": 10000, "cache_write": 0}),
        ]
        totals = compute_cache_waste(messages)
        assert totals["missCount"] == 1
        assert totals["missedTokens"] == 20000
