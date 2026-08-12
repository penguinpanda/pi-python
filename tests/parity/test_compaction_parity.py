"""Compaction parity golden 测试。"""

from __future__ import annotations

from pi_agent.compaction import CompactionSettings
from pi_agent.compaction_utils import should_compact


def test_should_compact_golden_threshold() -> None:
    settings = CompactionSettings(enabled=True, reserve_tokens=100)
    assert should_compact(901, 1000, settings) is True
    assert should_compact(900, 1000, settings) is False
    assert (
        should_compact(10000, 1000, CompactionSettings(enabled=False, reserve_tokens=100)) is False
    )
