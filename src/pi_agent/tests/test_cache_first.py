"""pi_agent.compaction_utils.apply_cache_first_truncation 单元测试。"""

from __future__ import annotations

from pi_agent.compaction_utils import (
    CACHE_FIRST_TRUNCATED_MARKER,
    TOOL_RESULT_MAX_CHARS,
    apply_cache_first_truncation,
)


def _tool_result(text: str, *, tool_call_id: str = "call-1") -> dict:
    return {
        "role": "toolResult",
        "tool_call_id": tool_call_id,
        "tool_name": "read",
        "content": [{"type": "text", "text": text}],
        "is_error": False,
        "timestamp": 1,
    }


def _bash(output: str) -> dict:
    return {
        "role": "bashExecution",
        "command": "npm install",
        "output": output,
        "exitCode": 0,
        "timestamp": 2,
    }


def test_large_tool_result_truncated_to_fixed_marker():
    msg = _tool_result("x" * (TOOL_RESULT_MAX_CHARS + 1))
    (result,) = apply_cache_first_truncation([msg])
    assert result["content"] == [{"type": "text", "text": CACHE_FIRST_TRUNCATED_MARKER}]
    assert result["tool_call_id"] == "call-1"
    assert result["timestamp"] == 1


def test_small_tool_result_unchanged():
    msg = _tool_result("ok")
    assert apply_cache_first_truncation([msg]) == [msg]


def test_exact_threshold_kept():
    msg = _tool_result("q" * TOOL_RESULT_MAX_CHARS)
    assert apply_cache_first_truncation([msg]) == [msg]


def test_large_bash_output_truncated():
    msg = _bash("y" * (TOOL_RESULT_MAX_CHARS + 1))
    (result,) = apply_cache_first_truncation([msg])
    assert result["output"] == CACHE_FIRST_TRUNCATED_MARKER
    assert result["command"] == "npm install"
    assert result["exitCode"] == 0


def test_user_assistant_toolcall_untouched():
    user = {"role": "user", "content": "u" * 5000, "timestamp": 1}
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "a" * 5000},
            {
                "type": "toolCall",
                "name": "read",
                "arguments": {"path": "x" * 5000},
                "id": "c",
            },
        ],
        "timestamp": 2,
    }
    assert apply_cache_first_truncation([user, assistant]) == [user, assistant]


def test_truncation_is_idempotent():
    msg = _tool_result("z" * (TOOL_RESULT_MAX_CHARS + 1))
    once = apply_cache_first_truncation([msg])
    twice = apply_cache_first_truncation(once)
    assert twice == once
    assert twice[0]["content"][0]["text"] == CACHE_FIRST_TRUNCATED_MARKER


def test_under_budget_no_truncation():
    msg = _tool_result("x" * 3000)
    assert apply_cache_first_truncation([msg], context_window=100000, reserve_tokens=0) == [msg]


def test_over_budget_truncates_tail_first():
    old = _tool_result("o" * 8000, tool_call_id="old")
    new = _tool_result("n" * 6000, tool_call_id="new")
    result = apply_cache_first_truncation([old, new], context_window=2500, reserve_tokens=0)
    assert result[0] == old
    assert result[1]["content"][0]["text"] == CACHE_FIRST_TRUNCATED_MARKER


def test_budget_exhausted_truncates_all():
    old = _tool_result("o" * 8000, tool_call_id="old")
    new = _tool_result("n" * 6000, tool_call_id="new")
    result = apply_cache_first_truncation([old, new], context_window=500, reserve_tokens=0)
    assert result[0]["content"][0]["text"] == CACHE_FIRST_TRUNCATED_MARKER
    assert result[1]["content"][0]["text"] == CACHE_FIRST_TRUNCATED_MARKER
