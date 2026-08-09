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


def test_large_tool_result_truncated_keeps_head_and_tail():
    msg = _tool_result("H" * 3000 + "M" * 5000 + "T" * 3000)
    (result,) = apply_cache_first_truncation([msg])
    text = result["content"][0]["text"]
    assert text.startswith(CACHE_FIRST_TRUNCATED_MARKER)
    assert "omitted" in text
    assert text.startswith(CACHE_FIRST_TRUNCATED_MARKER + "\n\n" + "H" * 3000)
    assert text.endswith("T" * 2000)
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
    assert result["output"].startswith(CACHE_FIRST_TRUNCATED_MARKER)
    assert "omitted" in result["output"]
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
    assert twice[0]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)


def test_under_budget_no_truncation():
    msg = _tool_result("x" * 3000)
    assert apply_cache_first_truncation([msg], context_window=100000, reserve_tokens=0) == [msg]


def test_over_budget_truncates_tail_first():
    old = _tool_result("o" * 8000, tool_call_id="old")
    new = _tool_result("n" * 20000, tool_call_id="new")
    result = apply_cache_first_truncation(
        [old, new], context_window=4600, reserve_tokens=0, protect_recent_tokens=0
    )
    assert result[0] == old
    assert result[1]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)


def test_budget_exhausted_truncates_all():
    old = _tool_result("o" * 8000, tool_call_id="old")
    new = _tool_result("n" * 6000, tool_call_id="new")
    result = apply_cache_first_truncation(
        [old, new], context_window=500, reserve_tokens=0, protect_recent_tokens=0
    )
    assert result[0]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)
    assert result[1]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)


def test_budget_counts_full_state_not_last_usage():
    """预算必须按完整 state 估算：前面的大工具输出不能被最后一条 usage 掩盖。"""
    large = _tool_result("x" * 20000, tool_call_id="large")
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input": 900,
            "output": 100,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 1000,
        },
        "timestamp": 3,
    }
    messages = [large, assistant]
    result = apply_cache_first_truncation(
        messages, context_window=3000, reserve_tokens=0, protect_recent_tokens=0
    )
    assert result[0]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)
    assert result[1] == assistant


def test_protected_recent_tail_not_truncated():
    old = _tool_result("o" * 20000, tool_call_id="old")
    recent = _tool_result("r" * 20000, tool_call_id="recent")
    result = apply_cache_first_truncation(
        [old, recent],
        context_window=3000,
        reserve_tokens=0,
        protect_recent_tokens=3000,
    )
    assert result[0]["content"][0]["text"].startswith(CACHE_FIRST_TRUNCATED_MARKER)
    assert result[1]["content"][0]["text"] == "r" * 20000


def test_archive_writes_original_before_truncation(tmp_path):
    text = "A" * (TOOL_RESULT_MAX_CHARS + 1)
    msg = _tool_result(text, tool_call_id="archived")
    archive_dir = tmp_path / "archive"
    (result,) = apply_cache_first_truncation([msg], archive_dir=archive_dir)
    files = list(archive_dir.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == text
    assert files[0].name in result["content"][0]["text"]
