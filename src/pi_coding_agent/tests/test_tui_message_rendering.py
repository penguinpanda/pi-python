"""TUI 消息渲染（P3：tool-execution / skill / 摘要条目）测试。"""

from __future__ import annotations

from pi_tui.components import message_to_entries


def test_tool_result_error_label():
    entries = message_to_entries(
        {
            "role": "toolResult",
            "tool_name": "bash",
            "is_error": True,
            "content": [{"type": "text", "text": "boom"}],
        }
    )
    assert entries == [("Tool: bash (error)", "boom")]


def test_tool_result_ok_label():
    entries = message_to_entries(
        {
            "role": "toolResult",
            "tool_name": "read",
            "content": [{"type": "text", "text": "content"}],
        }
    )
    assert entries == [("Tool: read", "content")]


def test_assistant_tool_calls_rendered_individually():
    entries = message_to_entries(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "toolCall", "name": "bash", "arguments": {"command": "ls"}},
                {"type": "toolCall", "name": "read", "arguments": {"path": "a.txt"}},
            ],
        }
    )
    labels = [label for label, _text in entries]
    assert labels == ["Assistant", "Tool call", "Tool call"]
    assert entries[1][1] == "bash({'command': 'ls'})"


def test_assistant_tool_calls_hidden_when_show_tools_false():
    entries = message_to_entries(
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "name": "bash", "arguments": {}},
                {"type": "text", "text": "done"},
            ],
        },
        show_tools=False,
    )
    assert entries == [("Assistant", "done")]


def test_compaction_summary_label():
    entries = message_to_entries(
        {
            "role": "compactionSummary",
            "summary": "earlier context",
        }
    )
    assert entries == [("Compaction summary", "earlier context")]


def test_branch_summary_label():
    entries = message_to_entries(
        {
            "role": "branchSummary",
            "summary": "branch context",
        }
    )
    assert entries == [("Branch summary", "branch context")]


def test_skill_invocation_label():
    entries = message_to_entries(
        {
            "role": "skillInvocation",
            "content": "Invoked skill: docs",
        }
    )
    assert entries == [("Skill", "Invoked skill: docs")]
