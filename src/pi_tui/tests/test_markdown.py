"""Markdown 渲染测试。"""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.markdown import Markdown

from pi_tui.markdown import label_icon, render_labeled_markdown


def _render(renderable) -> str:
    console = Console(record=True, width=80, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def _find_markdown(renderable):
    if isinstance(renderable, Markdown):
        return renderable
    nested = getattr(renderable, "renderables", None) or getattr(renderable, "renderable", None)
    if isinstance(nested, (list, tuple)):
        for item in nested:
            found = _find_markdown(item)
            if found is not None:
                return found
    elif nested is not None:
        return _find_markdown(nested)
    return None


def test_heading_and_bold() -> None:
    out = _render(render_labeled_markdown("Assistant", "# Title\n\n**bold** text"))
    assert "Assistant" in out
    assert "Title" in out
    assert "bold" in out


def test_code_block() -> None:
    out = _render(render_labeled_markdown("Assistant", "```python\nprint(1)\n```"))
    assert "print(1)" in out


def test_list() -> None:
    out = _render(render_labeled_markdown("Assistant", "- item one\n- item two"))
    assert "item one" in out
    assert "item two" in out


def test_table() -> None:
    md = "| name | value |\n| --- | --- |\n| a | 1 |"
    out = _render(render_labeled_markdown("Assistant", md))
    assert "name" in out
    assert "value" in out
    assert "a" in out


def test_link() -> None:
    out = _render(render_labeled_markdown("Assistant", "[pi](https://example.com)"))
    assert "pi" in out


def test_rich_markup_passthrough() -> None:
    out = _render(render_labeled_markdown("Assistant", "[b]hi[/b]"))
    assert "hi" in out
    assert "[/" not in out


def test_brackets_are_escaped() -> None:
    out = _render(render_labeled_markdown("Assistant", "a [b] c"))
    assert "a [b] c" in out


def test_label_icons() -> None:
    user_out = _render(render_labeled_markdown("User", "hi"))
    assistant_out = _render(render_labeled_markdown("Assistant", "hi"))
    tool_out = _render(render_labeled_markdown("Tool: bash", "out"))
    assert "👤 User" in user_out
    assert "🤖 Assistant" in assistant_out
    assert "🛠 Tool: bash" in tool_out
    assert label_icon("Compaction summary") == "📦"
    assert label_icon("Unknown") == "▸"


def test_speaking_indicator() -> None:
    out = _render(render_labeled_markdown("Assistant", "hi", speaking=True))
    assert "Speaking" in out
    assert "🤖 Assistant" in out


def test_first_line_indent() -> None:
    out = _render(render_labeled_markdown("Assistant", "first line\n\nsecond line"))
    assert "\n  first line" in out
    assert "\n  second line" in out


@pytest.mark.asyncio
async def test_message_entry_renders_markdown() -> None:
    from textual.app import App

    from rich.console import Group

    from pi_tui.components import MessageEntry

    class Host(App):
        def compose(self):
            yield MessageEntry("Assistant", "# Hi\n\n- item")

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        entry = app.query_one(MessageEntry)
        assert entry.label == "Assistant"
        assert entry.entry_text == "# Hi\n\n- item"
        content = entry.content
        assert isinstance(content, Group)
        assert _find_markdown(content) is not None


@pytest.mark.asyncio
async def test_message_entry_set_text_updates() -> None:
    from textual.app import App

    from pi_tui.components import MessageEntry

    class Host(App):
        def compose(self):
            yield MessageEntry("Assistant", "first")

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        entry = app.query_one(MessageEntry)
        entry.set_text("second")
        assert entry.entry_text == "second"
        assert entry.content is not None


@pytest.mark.asyncio
async def test_message_entry_set_speaking_updates() -> None:
    from textual.app import App

    from pi_tui.components import MessageEntry

    class Host(App):
        def compose(self):
            yield MessageEntry("Assistant", "hi")

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        entry = app.query_one(MessageEntry)
        entry.set_speaking(True)
        label_text = entry.content.renderables[0]
        assert "Speaking" in label_text.plain
        entry.set_speaking(False)
        label_text = entry.content.renderables[0]
        assert "Speaking" not in label_text.plain
