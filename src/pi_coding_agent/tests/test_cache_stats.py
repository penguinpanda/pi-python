"""Prompt 缓存浪费统计测试（对齐 TS cache-stats.ts）。"""

from __future__ import annotations

from pi_coding_agent.cache_stats import (
    CACHE_TTL_MS,
    compute_cache_waste,
    detect_recent_cache_miss,
    estimate_cache_state,
)


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


class TestDetectRecentCacheMiss:
    def test_single_request_returns_none(self):
        messages = [
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
        ]
        assert detect_recent_cache_miss(messages) is None

    def test_large_miss_above_token_threshold(self):
        messages = [
            _assistant(usage={"input": 30000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 20000, "cache_read": 0, "cache_write": 0}),
        ]
        miss = detect_recent_cache_miss(messages)
        assert miss is not None
        assert miss["missedTokens"] == 20000
        assert miss["missCount"] == 1

    def test_large_miss_above_cost_threshold_with_small_tokens(self):
        messages = [
            _assistant(usage={"input": 3000, "cache_read": 0, "cache_write": 0}),
            _assistant(
                usage={
                    "input": 3000,
                    "cache_read": 1500,
                    "cache_write": 0,
                    "cost": {
                        "input": 1.0,
                        "output": 0.0,
                        "cache_read": 0.0,
                        "cache_write": 0.0,
                        "total": 1.0,
                    },
                }
            ),
        ]
        miss = detect_recent_cache_miss(messages)
        assert miss is not None
        assert miss["missedCost"] >= 0.1

    def test_small_miss_ignored(self):
        messages = [
            _assistant(usage={"input": 5000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 4000, "cache_read": 4000, "cache_write": 0}),
        ]
        assert detect_recent_cache_miss(messages) is None

    def test_miss_below_noise_floor_ignored(self):
        messages = [
            _assistant(usage={"input": 2000, "cache_read": 0, "cache_write": 0}),
            _assistant(usage={"input": 1500, "cache_read": 500, "cache_write": 0}),
        ]
        assert detect_recent_cache_miss(messages) is None


class TestEstimateCacheState:
    def test_unknown_without_usage(self):
        assert estimate_cache_state([{"role": "user", "content": "hi"}]) == "unknown"

    def test_warm_within_ttl(self):
        now = 1_000_000
        messages = [
            {
                "role": "assistant",
                "content": [],
                "usage": {"input": 1, "cache_read": 0, "cache_write": 0},
                "timestamp": now - 1000,
            }
        ]
        assert estimate_cache_state(messages, now_ms=now) == "warm"

    def test_cold_after_ttl(self):
        now = 1_000_000
        messages = [
            {
                "role": "assistant",
                "content": [],
                "usage": {"input": 1, "cache_read": 0, "cache_write": 0},
                "timestamp": now - CACHE_TTL_MS - 1000,
            }
        ]
        assert estimate_cache_state(messages, now_ms=now) == "cold"

    def test_unknown_when_timestamp_missing(self):
        messages = [
            {
                "role": "assistant",
                "content": [],
                "usage": {"input": 1, "cache_read": 0, "cache_write": 0},
            }
        ]
        assert estimate_cache_state(messages, now_ms=1_000_000) == "unknown"
