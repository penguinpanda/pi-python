"""TreeSelector 树过滤模式测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Label, Static

from pi_tui.selectors import (
    TreeSelector,
    _flatten_tree,
    _node_copy_text,
    format_label_timestamp,
    node_passes_tree_filter,
)


def _node(entry=None, label=None, children=None, label_timestamp=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(id(entry or label)),
        entry=entry,
        label=label,
        children=children or [],
        label_timestamp=label_timestamp,
    )


def test_default_hides_settings_entries() -> None:
    assert node_passes_tree_filter(_node({"type": "message", "role": "user"}), "default")
    assert not node_passes_tree_filter(_node({"type": "label"}), "default")
    assert not node_passes_tree_filter(_node({"type": "model_change"}), "default")
    assert not node_passes_tree_filter(_node({"type": "session_info"}), "default")


def test_no_tools_hides_tool_results() -> None:
    assert not node_passes_tree_filter(_node({"type": "message", "role": "toolResult"}), "no-tools")
    assert node_passes_tree_filter(_node({"type": "message", "role": "user"}), "no-tools")
    assert not node_passes_tree_filter(_node({"type": "label"}), "no-tools")


def test_user_only() -> None:
    assert node_passes_tree_filter(_node({"type": "message", "role": "user"}), "user-only")
    assert not node_passes_tree_filter(_node({"type": "message", "role": "assistant"}), "user-only")


def test_labeled_only() -> None:
    assert node_passes_tree_filter(
        _node({"type": "message", "role": "user"}, label="x"), "labeled-only"
    )
    assert not node_passes_tree_filter(_node({"type": "message", "role": "user"}), "labeled-only")


def test_all_shows_everything() -> None:
    assert node_passes_tree_filter(_node({"type": "label"}), "all")
    assert node_passes_tree_filter(_node({"type": "model_change"}), "all")


def test_format_label_timestamp() -> None:
    formatted = format_label_timestamp("2026-08-05T12:34:56+00:00")
    assert len(formatted) == 8
    assert formatted[2] == ":" and formatted[5] == ":"
    assert format_label_timestamp("not-a-date") == "not-a-date"
    assert format_label_timestamp("") == ""


def test_flatten_tree_with_label_timestamp() -> None:
    node = _node(
        {"type": "message", "role": "user"},
        label="u1",
        label_timestamp="2026-08-05T12:00:00+00:00",
    )
    rows = _flatten_tree([node], None, show_label_timestamps=True)
    assert "@" in rows[0][2]
    rows_off = _flatten_tree([node], None, show_label_timestamps=False)
    assert "@" not in rows_off[0][2]


def test_node_copy_text_returns_full_message() -> None:
    assert (
        _node_copy_text(
            _node(
                {
                    "type": "message",
                    "role": "user",
                    "message": {"role": "user", "content": "first\nsecond"},
                }
            )
        )
        == "first\nsecond"
    )
    list_content = _node_copy_text(
        _node(
            {
                "type": "message",
                "role": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                },
            }
        )
    )
    assert list_content == "a\nb"
    assert _node_copy_text(_node({"type": "label"}, label="lab")) == "lab"
    assert _node_copy_text(_node({"type": "label"})) == ""


class _Host(App):
    def __init__(self, tree) -> None:
        super().__init__()
        self._tree = tree

    def compose(self):
        yield Static("")
        yield TreeSelector(self._tree)

    def on_mount(self) -> None:
        self.query_one(TreeSelector).query_one("#tree-list").focus()


@pytest.mark.asyncio
async def test_tree_selector_filter_cycles() -> None:
    tree = [
        _node({"type": "message", "role": "user"}, label="u1"),
        _node({"type": "message", "role": "toolResult"}),
        _node({"type": "label"}),
        _node({"type": "model_change"}),
    ]
    app = _Host(tree)
    async with app.run_test() as pilot:
        await pilot.pause()
        selector = app.query_one(TreeSelector)
        assert selector.filter_mode == "default"
        assert len(selector.query_one("#tree-list").children) == 2

        await pilot.press("f")
        await pilot.pause()
        assert selector.filter_mode == "no-tools"
        assert "[no-tools" in selector.query_one(Label).render().plain
        assert len(selector.query_one("#tree-list").children) == 1

        await pilot.press("f")
        await pilot.pause()
        assert selector.filter_mode == "user-only"
        assert len(selector.query_one("#tree-list").children) == 1

        await pilot.press("f")
        await pilot.pause()
        assert selector.filter_mode == "labeled-only"
        assert len(selector.query_one("#tree-list").children) == 1

        await pilot.press("f")
        await pilot.pause()
        assert selector.filter_mode == "all"
        assert len(selector.query_one("#tree-list").children) == 4


@pytest.mark.asyncio
async def test_tree_selector_toggles_label_timestamps() -> None:
    tree = [
        _node(
            {"type": "message", "role": "user"},
            label="u1",
            label_timestamp="2026-08-05T12:00:00+00:00",
        )
    ]
    app = _Host(tree)
    async with app.run_test() as pilot:
        await pilot.pause()
        selector = app.query_one(TreeSelector)
        assert selector.show_label_timestamps is False
        row_label = selector.query_one("#tree-list").children[0].query_one(Label)
        assert "@" not in row_label.render().plain

        await pilot.press("t")
        await pilot.pause()
        assert selector.show_label_timestamps is True
        assert "[+label time" in selector.query_one(Label).render().plain
        row_label = selector.query_one("#tree-list").children[0].query_one(Label)
        assert "@" in row_label.render().plain

        await pilot.press("t")
        await pilot.pause()
        assert selector.show_label_timestamps is False


class _CopyHost(App):
    def __init__(self, tree) -> None:
        super().__init__()
        self._tree = tree
        self.copied: list[str] = []

    def compose(self):
        yield Static("")
        yield TreeSelector(self._tree)

    def on_mount(self) -> None:
        self.query_one(TreeSelector).query_one("#tree-list").focus()

    def on_copy_requested(self, message) -> None:
        self.copied.append(message.text)


@pytest.mark.asyncio
async def test_tree_selector_copy_selected_posts_message() -> None:
    tree = [
        _node(
            {
                "type": "message",
                "role": "user",
                "message": {"role": "user", "content": "first\nsecond"},
            },
            label="snippet",
        )
    ]
    app = _CopyHost(tree)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert app.copied == ["first\nsecond"]


@pytest.mark.asyncio
async def test_choice_selector_copy_selected_posts_message() -> None:
    from pi_tui.selectors import ChoiceSelector

    class _ChoiceHost(App):
        def __init__(self) -> None:
            super().__init__()
            self.copied: list[str] = []

        def compose(self):
            yield ChoiceSelector("Pick", ["alpha", "beta"], current="alpha")

        def on_copy_requested(self, message) -> None:
            self.copied.append(message.text)

    app = _ChoiceHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert app.copied == ["alpha"]
