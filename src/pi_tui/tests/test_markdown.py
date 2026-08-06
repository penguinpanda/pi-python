"""引擎版 Markdown / 消息条目渲染测试。"""

from __future__ import annotations

from pi_tui.components import MessageEntry, _render_labeled_markdown
from pi_tui.engine.text import render_markdown, strip_ansi
from pi_tui.markdown import label_icon


def _plain(lines) -> list[str]:
    return [strip_ansi(line.text()) for line in lines]


def test_heading_and_bold() -> None:
    lines = render_markdown("# Title\n\n**bold** text", 40)
    plain = _plain(lines)
    assert any("Title" in line for line in plain)
    assert any("bold" in line for line in plain)


def test_code_block() -> None:
    lines = render_markdown("```python\nprint(1)\n```", 40)
    plain = _plain(lines)
    assert any("print(1)" in line for line in plain)


def test_list() -> None:
    lines = render_markdown("- one\n- two", 40)
    plain = _plain(lines)
    assert any("one" in line for line in plain)
    assert any("two" in line for line in plain)


def test_table() -> None:
    lines = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |", 40)
    plain = _plain(lines)
    assert any("a" in line and "b" in line for line in plain)
    assert any("1" in line and "2" in line for line in plain)


def test_link() -> None:
    lines = render_markdown("[pi](https://example.com)", 40)
    plain = _plain(lines)
    assert any("pi" in line for line in plain)


def test_label_icons() -> None:
    assert label_icon("User") == "👤"
    assert label_icon("Assistant") == "🤖"
    assert label_icon("Tool: bash") == "🛠"
    assert label_icon("System") == "⚙️"


def test_speaking_indicator() -> None:
    lines = _render_labeled_markdown("Assistant", "hi", 40, speaking=True)
    assert "Speaking" in lines[0].text()


def test_first_line_indent() -> None:
    lines = _render_labeled_markdown("User", "hello", 40)
    assert lines[0].text().startswith("👤 User")
    assert lines[1].text().startswith("  hello")


def test_message_entry_renders_markdown() -> None:
    entry = MessageEntry("Assistant", "**bold** body")
    lines = entry.render(40, 10)
    plain = _plain(lines)
    assert any("bold" in line for line in plain)


def test_message_entry_set_text_updates() -> None:
    entry = MessageEntry("User", "old")
    entry.set_text("new **content**")
    plain = _plain(entry.render(40, 10))
    assert any("new" in line and "content" in line for line in plain)


def test_message_entry_set_speaking_updates() -> None:
    entry = MessageEntry("Assistant", "hello")
    entry.set_speaking(True)
    assert "Speaking" in entry.render(40, 10)[0].text()
