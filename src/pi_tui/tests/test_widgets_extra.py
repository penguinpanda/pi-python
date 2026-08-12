"""engine/widgets 补充测试。"""

from __future__ import annotations

from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import (
    Container,
    Editor,
    Input,
    SelectList,
    Static,
    Widget,
)


def _key(name: str, char: str | None = None) -> Key:
    return Key(name=name, char=char)


def test_widget_base_methods() -> None:
    widget = Widget(id="w")
    assert widget.cursor_position() is None
    assert widget.handle_key(_key("x", "x")) is False
    assert widget.handle_mouse(None) is False
    assert widget.content_size() == (0, 0)
    assert widget.natural_size(80) == (0, 0)
    widget.focus()
    widget.blur()
    widget.post_message(type("M", (), {})())
    assert "w" in repr(widget)


def test_container_children_and_layout() -> None:
    container = Container(direction="vertical")
    a = Static("a", height=1, id="a")
    b = Static("b", height=1, id="b")
    container.mount(a)
    container.mount(b)
    assert list(container.walk()) == [container, a, b]
    assert container.query_one("a") == a
    assert container.query_all(Static) == [a, b]
    assert container.find(lambda widget: widget is b) == [b]
    container.set_child_basis(a, 3)
    assert container.layout_node().entries[0].basis == 3
    container.remove(a)
    assert list(container.walk()) == [container, b]
    container.clear()
    assert list(container.walk()) == [container]
    assert container.natural_size(80) == (80, 0)


def test_input_navigation_and_editing() -> None:
    field = Input(value="abc")
    field.cursor = 1
    assert field.handle_key(_key("left")) is True
    assert field.cursor == 0
    assert field.handle_key(_key("right")) is True
    assert field.cursor == 1
    assert field.handle_key(_key("end")) is True
    assert field.cursor == 3
    assert field.handle_key(_key("home")) is True
    assert field.handle_key(_key("ctrl+f")) is True
    assert field.handle_key(_key("delete")) is True
    assert field.value == "ac"
    assert field.handle_key(_key("ctrl+a")) is True
    field.cursor = 1
    assert field.handle_key(_key("ctrl+u")) is True
    assert field.value == "c"
    assert field.handle_key(_key("d", "d")) is True
    assert field.value == "dc"
    lines = field.render(80, 1)
    assert "dc" in lines[0].text()


def test_editor_insert_undo_redo_selection() -> None:
    editor = Editor(text="hello")
    editor.cursor_col = 5
    editor.insert("\nworld")
    assert editor.text == "hello\nworld"
    editor.cursor_row = 0
    editor.cursor_col = 0
    editor.selection_anchor = (0, 0)
    editor.cursor_row = 1
    editor.cursor_col = 5
    assert editor.selected_text == "hello\nworld"
    editor.delete(editor._selection_bounds()[0], editor._selection_bounds()[1])
    assert editor.text == ""
    editor.undo()
    assert editor.text == "hello\nworld"
    editor.redo()
    assert editor.text == ""
    editor.set_autocomplete_provider(object())
    assert editor.get_expanded_text() == ""
    editor.insert_text_at_cursor("x")
    assert editor.text == "x"


def test_select_list_navigation_and_filter() -> None:
    listing = SelectList(["a", "b", "c"])
    assert listing.handle_key(_key("down")) is True
    assert listing.selected_index == 1
    assert listing.handle_key(_key("up")) is True
    assert listing.selected_index == 0
    assert listing.handle_key(_key("b", "b")) is True
    assert [item.display_label for item in listing.filtered_items] == ["b"]
    listing.query = ""
    listing._apply_filter()
    assert len(listing.filtered_items) == 3
