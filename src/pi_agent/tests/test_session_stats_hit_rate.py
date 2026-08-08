"""pi_agent SessionStats.hitRate 单元测试。"""

from __future__ import annotations

from pi_agent.session.session import _get_session_stats


def _entry(usage: dict) -> dict:
    return {
        "type": "message",
        "id": "1",
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "message": {
            "role": "assistant",
            "content": [],
            "usage": usage,
        },
    }


def test_hit_rate_computed():
    stats = _get_session_stats(
        [
            _entry(
                {
                    "input": 100,
                    "output": 10,
                    "cacheRead": 50,
                    "cacheWrite": 0,
                    "cost": {"total": 0.001},
                }
            )
        ]
    )
    assert stats["hitRate"] == 50 / 150


def test_hit_rate_none_without_tokens():
    stats = _get_session_stats(
        [
            _entry(
                {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "cost": {"total": 0.0},
                }
            )
        ]
    )
    assert stats["hitRate"] is None
