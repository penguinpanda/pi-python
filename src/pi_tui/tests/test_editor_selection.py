"""编辑器选区与 kill ring 测试。"""

from __future__ import annotations

from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import Editor


def _key(name: str) -> Key:
    return Key(name)


def test_shift_selection_selects_text() -> None:
    editor = Editor(text="hello world")
    for _ in range(5):
        editor.handle_key(_key("shift+right"))
    assert editor.cursor_col == 5
    assert editor.selected_text == "hello"
    assert editor.selection == ((0, 0), (0, 5))


def test_plain_move_clears_selection() -> None:
    editor = Editor(text="hello")
    editor.handle_key(_key("shift+right"))
    assert editor.selected_text == "h"
    editor.handle_key(_key("left"))
    assert editor.selected_text == ""
    assert editor.selection is None


def test_multiline_selection() -> None:
    editor = Editor(text="a\nb\nc")
    editor.handle_key(_key("shift+down"))
    assert editor.selected_text == "a\n"


def test_kill_ring_kill_and_yank() -> None:
    editor = Editor(text="hello world")
    for _ in range(5):
        editor.handle_key(_key("shift+right"))
    editor.handle_key(_key("ctrl+k"))
    assert editor.text == "hello"
    assert editor.kill_ring == [" world"]
    editor.handle_key(_key("ctrl+y"))
    assert editor.text == "hello world"


def test_kill_at_line_end_joins_next_line() -> None:
    editor = Editor(text="a\nb")
    editor.move_cursor((0, 1))
    editor.handle_key(_key("ctrl+k"))
    assert editor.text == "ab"
    assert editor.kill_ring == [""]


def test_word_selection() -> None:
    editor = Editor(text="hello world")
    editor.handle_key(_key("ctrl+shift+right"))
    assert editor.selected_text == "hello"
    editor.handle_key(_key("ctrl+shift+right"))
    assert editor.selected_text == "hello world"


def test_ctrl_c_copies_selection() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    editor = Editor(text="hello world")
    editor.app = app
    editor.handle_key(_key("ctrl+shift+right"))
    editor.handle_key(_key("ctrl+c"))
    assert app.terminal.clipboard == ["hello"]
    assert editor.text == "hello world"


def test_ctrl_c_clears_without_selection() -> None:
    editor = Editor(text="hello")
    editor.handle_key(_key("ctrl+c"))
    assert editor.text == ""
