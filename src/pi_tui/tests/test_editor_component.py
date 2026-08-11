"""EditorComponent Protocol 测试。"""

from __future__ import annotations

from pi_tui.engine import EditorComponent
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import Editor


def test_editor_implements_editor_component() -> None:
    assert isinstance(Editor(), EditorComponent)


def test_editor_component_methods() -> None:
    editor: EditorComponent = Editor(text="hello")
    assert editor.get_text() == "hello"
    editor.set_text("world")
    assert editor.get_text() == "world"
    editor.insert_text_at_cursor("!")
    assert editor.get_expanded_text() == "world!"
    editor.add_to_history("world!")
    assert editor.handle_key(Key(name="escape")) is False
