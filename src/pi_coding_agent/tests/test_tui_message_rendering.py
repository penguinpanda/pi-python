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


def test_bash_execution_rendering():
    entries = message_to_entries(
        {
            "role": "bashExecution",
            "command": "npm run lint",
            "output": "All checks passed",
            "exitCode": 0,
            "cancelled": False,
            "truncated": False,
        }
    )
    assert entries == [("Bash", "$ npm run lint\nAll checks passed")]


def test_bash_execution_excluded_label_and_status():
    entries = message_to_entries(
        {
            "role": "bashExecution",
            "command": "ls",
            "output": "a.txt",
            "exitCode": 2,
            "cancelled": False,
            "truncated": True,
            "fullOutputPath": "/tmp/pi-bash.log",
            "excludeFromContext": True,
        }
    )
    assert entries == [
        (
            "Bash (excluded)",
            "$ ls\na.txt\n(exit 2) Output truncated. Full output: /tmp/pi-bash.log",
        )
    ]


def test_bash_execution_cancelled_rendering():
    entries = message_to_entries(
        {
            "role": "bashExecution",
            "command": "sleep 10",
            "output": "",
            "exitCode": None,
            "cancelled": True,
            "truncated": False,
        }
    )
    assert entries == [("Bash", "$ sleep 10\n(no output)\n(cancelled)")]


def test_custom_message_custom_renderer():
    message = {
        "role": "custom",
        "customType": "note",
        "content": [{"type": "text", "text": "payload"}],
    }
    entries = message_to_entries(
        message,
        custom_renderer=lambda m: f"RENDERED:{m.get('customType')}",
    )
    assert entries == [("note", "RENDERED:note")]
    # 无渲染器时回退到文本。
    fallback = message_to_entries(message)
    assert fallback == [("note", "payload")]


def test_markdown_transformers_chain():
    def upper(text, ctx):
        assert ctx["messageType"] == "user"
        return text.upper()

    def broken(text, ctx):
        raise RuntimeError("boom")

    def decorate(text, ctx):
        return f"*{text}*"

    entries = message_to_entries(
        {"role": "user", "content": "hi"},
        markdown_transformers=[upper, broken, decorate],
    )
    # 失败的变换器被跳过，保留上一步结果。
    assert entries == [("User", "*HI*")]


def test_hidden_thinking_label():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "reasoning"}],
    }
    default_entries = message_to_entries(message)
    assert default_entries == [("Thinking", "reasoning")]
    custom_entries = message_to_entries(message, hidden_thinking_label="Pondering...")
    assert custom_entries == [("Pondering...", "reasoning")]


def test_tool_renderer():
    message = {
        "role": "toolResult",
        "tool_name": "read",
        "content": [{"type": "text", "text": "line1\nline2"}],
    }
    default_entries = message_to_entries(message)
    assert default_entries[0][1] == "line1\nline2"
    custom_entries = message_to_entries(
        message,
        tool_renderer=lambda m: (
            "read: "
            + str(
                len(
                    "".join(
                        block.get("text", "")
                        for block in m.get("content") or []
                        if isinstance(block, dict)
                    ).splitlines()
                )
            )
            + " lines"
        ),
    )
    assert custom_entries == [("Tool: read", "read: 2 lines")]
