"""compaction 公共工具（compaction_utils.py）单元测试。"""

from __future__ import annotations

import pytest

from pi_agent.compaction_utils import (
    compute_file_lists,
    create_file_ops,
    estimate_tokens,
    extract_file_ops_from_message,
    format_file_operations,
    safe_json_stringify,
    serialize_conversation,
    truncate_for_summary,
)


def test_extract_file_ops_from_message_variants():
    ops = create_file_ops()
    extract_file_ops_from_message(
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "name": "read", "arguments": {"path": "a.py"}},
                {"type": "toolCall", "name": "write", "arguments": {"path": "b.py"}},
                {"type": "toolCall", "name": "edit", "arguments": {"path": "c.py"}},
                {"type": "toolCall", "name": "read", "arguments": {"path": "b.py"}},
                {"type": "toolCall", "name": "bash", "arguments": {}},
                {"type": "toolCall", "name": "read", "arguments": "not-a-dict"},
            ],
        },
        ops,
    )
    assert ops["read"] == {"a.py", "b.py"}
    assert ops["written"] == {"b.py"}
    assert ops["edited"] == {"c.py"}

    extract_file_ops_from_message({"role": "user", "content": []}, ops)
    assert ops["read"] == {"a.py", "b.py"}


def test_compute_and_format_file_lists():
    read_only, modified = compute_file_lists(
        {"read": {"a.py", "b.py"}, "written": {"b.py"}, "edited": {"c.py"}}
    )
    assert read_only == ["a.py"]
    assert modified == ["b.py", "c.py"]
    assert format_file_operations([], []) == ""
    assert "<read-files>" in format_file_operations(["a.py"], [])
    assert "<modified-files>" in format_file_operations([], ["b.py"])


def test_safe_json_stringify():
    assert safe_json_stringify({"a": 1}) == '{"a": 1}'
    assert safe_json_stringify(object()) == "[unserializable]"


def test_truncate_for_summary():
    assert truncate_for_summary("short", 10) == "short"
    out = truncate_for_summary("x" * 20, 5)
    assert "15 more characters truncated" in out


def test_serialize_conversation_roles():
    messages = [
        {"role": "user", "content": "hi", "timestamp": 1},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "answer"},
                {"type": "toolCall", "name": "read", "arguments": {"path": "a.py"}},
            ],
            "timestamp": 2,
        },
        {"role": "toolResult", "content": [{"type": "text", "text": "out"}], "timestamp": 3},
        {
            "role": "bashExecution",
            "command": "ls",
            "output": "x",
            "exitCode": 1,
            "timestamp": 4,
        },
    ]
    text = serialize_conversation(messages)
    assert "[User]: hi" in text
    assert "[Assistant thinking]: think" in text
    assert "[Assistant]: answer" in text
    assert 'read(path="a.py")' in text
    assert "[Tool result]: out" in text
    assert "[Bash]: ls (exit 1)" in text


def test_estimate_tokens_counts_blocks():
    user = {"role": "user", "content": "x" * 8, "timestamp": 1}
    assert estimate_tokens(user) == 2
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "a" * 8},
            {"type": "toolCall", "name": "read", "arguments": {"path": "x"}},
        ],
        "timestamp": 2,
    }
    assert estimate_tokens(assistant) > 0
    assert estimate_tokens({"role": "other", "content": [], "timestamp": 3}) == 0


@pytest.mark.asyncio
async def test_branch_summary_collect_missing_entry_raises():
    from pi_agent.branch_summarization import collect_entries_for_branch_summary
    from pi_agent.session.v4.types import SessionError

    class _FakeSession:
        async def get_branch(self, leaf_id):
            if leaf_id == "old":
                return [{"id": "old"}, {"id": "root"}]
            return [{"id": "target"}, {"id": "root"}]

        async def get_entry(self, entry_id):
            return None

    with pytest.raises(SessionError, match="not found"):
        await collect_entries_for_branch_summary(_FakeSession(), "old", "target")
