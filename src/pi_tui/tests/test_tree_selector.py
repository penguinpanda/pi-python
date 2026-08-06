"""引擎版 TreeSelector / 树过滤测试。"""

from __future__ import annotations

from pi_coding_agent._session_manager import SessionTreeNode
from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.keys import Key
from pi_tui.selectors import (
    ChoiceSelector,
    TreeSelector,
    _flatten_tree,
    _node_copy_text,
    format_label_timestamp,
    node_passes_tree_filter,
)


def _node(
    entry: dict, label: str | None = None, children=None, node_id: str = "n1"
) -> SessionTreeNode:
    return SessionTreeNode(
        id=node_id,
        parent_id=None,
        entry=entry,
        label=label,
        children=children or [],
    )


def _key(name: str) -> Key:
    return Key(name)


def test_default_hides_settings_entries() -> None:
    node = _node({"type": "label"})
    assert node_passes_tree_filter(node, "default") is False
    message = _node({"type": "message", "role": "user"})
    assert node_passes_tree_filter(message, "default") is True


def test_no_tools_hides_tool_results() -> None:
    node = _node({"type": "message", "role": "toolResult"})
    assert node_passes_tree_filter(node, "no-tools") is False


def test_user_only() -> None:
    user = _node({"type": "message", "role": "user"})
    assistant = _node({"type": "message", "role": "assistant"})
    assert node_passes_tree_filter(user, "user-only") is True
    assert node_passes_tree_filter(assistant, "user-only") is False


def test_labeled_only() -> None:
    assert node_passes_tree_filter(_node({"type": "message"}, label="x"), "labeled-only") is True
    assert node_passes_tree_filter(_node({"type": "message"}), "labeled-only") is False


def test_all_shows_everything() -> None:
    assert node_passes_tree_filter(_node({"type": "label"}), "all") is True


def test_format_label_timestamp() -> None:
    assert format_label_timestamp(None) == ""
    assert format_label_timestamp("bad-value") == "bad-value"
    parsed = format_label_timestamp("2026-08-05T10:20:30Z")
    assert ":" in parsed


def test_flatten_tree_with_label_timestamp() -> None:
    child = _node({"type": "message"}, label="child", node_id="c1")
    parent = _node({"type": "message"}, label="parent", node_id="p1", children=[child])
    rows = _flatten_tree([parent], None)
    assert len(rows) == 2
    assert rows[0][1] == ""
    assert rows[1][1] == "└─"


def test_node_copy_text_returns_full_message() -> None:
    node = _node({"type": "message", "message": {"content": "full text"}})
    assert _node_copy_text(node) == "full text"


def test_tree_selector_filter_cycles() -> None:
    selector = TreeSelector([_node({"type": "label"}), _node({"type": "message"})])
    assert selector.filter_mode == "default"
    selector.handle_key(_key("f"))
    assert selector.filter_mode == "no-tools"
    selector.handle_key(_key("f"))
    assert selector.filter_mode == "user-only"


def test_tree_selector_toggles_label_timestamps() -> None:
    selector = TreeSelector([_node({"type": "message"})])
    assert selector.show_label_timestamps is False
    selector.handle_key(_key("t"))
    assert selector.show_label_timestamps is True


def test_tree_selector_copy_selected_posts_message() -> None:
    app = App(terminal=FakeTerminal(size=(60, 20)))
    copied: list[str] = []
    app.on_copy_requested = lambda message: copied.append(message.text)  # type: ignore[method-assign]
    selector = TreeSelector(
        [_node({"type": "message", "message": {"content": "copy me"}}, label="x")]
    )
    selector.app = app
    selector.handle_key(_key("enter"))
    assert copied == ["copy me"]


def test_choice_selector_copy_selected_posts_message() -> None:
    app = App(terminal=FakeTerminal(size=(60, 20)))
    copied: list[str] = []
    app.on_copy_requested = lambda message: copied.append(message.text)  # type: ignore[method-assign]
    selector = ChoiceSelector("Pick", ["alpha", "beta"])
    selector.app = app
    selector.handle_key(_key("enter"))
    assert copied == ["alpha"]
