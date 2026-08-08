"""Long-session cache-first eval 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from pi_evals.long_session_cache_eval import (
    BIG_FILE_LINES,
    STEPS,
    _big_line,
    _long_session_judge,
    _setup_workspace,
)


def _ctx(output):
    return SimpleNamespace(output=output)


def test_judge_passes_complete_session():
    output = {
        "response": "line-0001: xxxxx",
        "assistantMessages": ["a"] * len(STEPS),
        "toolCalls": 6,
    }
    result = _long_session_judge(_ctx(output))
    assert result["score"] == 1


def test_judge_fails_missing_steps():
    output = {
        "response": "line-0001: xxxxx",
        "assistantMessages": ["a"],
        "toolCalls": 6,
    }
    result = _long_session_judge(_ctx(output))
    assert result["score"] == 0
    assert any("not all steps" in f for f in result["metadata"]["rationale"].split("; "))


def test_judge_fails_missing_final_lines():
    output = {
        "response": "42",
        "assistantMessages": ["a"] * len(STEPS),
        "toolCalls": 6,
    }
    result = _long_session_judge(_ctx(output))
    assert result["score"] == 0
    assert any("final response" in f for f in result["metadata"]["rationale"].split("; "))


def test_workspace_setup_writes_large_file(tmp_path):
    _setup_workspace(tmp_path)
    big = tmp_path / "data" / "big.txt"
    assert big.exists()
    assert len(big.read_text(encoding="utf-8").splitlines()) == BIG_FILE_LINES
    assert len(_big_line(1)) > 2000 or BIG_FILE_LINES > 100
    assert (tmp_path / "data" / "target.txt").read_text(encoding="utf-8") == "ANSWER=42\n"
