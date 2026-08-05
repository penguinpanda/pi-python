"""Markdown 渲染测试。"""

from __future__ import annotations

import pytest
from rich.console import Console

from pi_tui.markdown import render_labeled_markdown


def _render(renderable) -> str:
    console = Console(record=True, width=80, force_terminal=False)
    console.print(renderable)
    return console.export_text()


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


@pytest.mark.asyncio
async def test_message_entry_renders_markdown() -> None:
    from textual.app import App

    from rich.console import Group
    from rich.markdown import Markdown

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
        assert any(isinstance(renderable, Markdown) for renderable in content.renderables)


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
