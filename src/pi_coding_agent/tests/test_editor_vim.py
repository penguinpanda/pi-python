"""引擎版 vim 编辑器测试。"""

from __future__ import annotations

from pi_coding_agent.modes.interactive.components import PiEditorVim
from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.keys import Key


def _editor(text: str = "") -> PiEditorVim:
    return PiEditorVim(text=text)


def _key(name: str) -> Key:
    return Key(name)


def test_escape_toggles_mode() -> None:
    editor = _editor("hello")
    assert editor.vim_mode == "insert"
    assert editor.handle_key(_key("escape"))
    assert editor.vim_mode == "normal"
    assert editor.handle_key(_key("escape"))
    assert editor.vim_mode == "insert"


def test_normal_mode_navigation_jk() -> None:
    editor = _editor("a\nb\nc")
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("j"))
    assert editor.cursor_row == 1
    editor.handle_key(_key("j"))
    assert editor.cursor_row == 2
    editor.handle_key(_key("k"))
    assert editor.cursor_row == 1


def test_dd_deletes_current_line() -> None:
    editor = _editor("a\nb\nc")
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("j"))
    editor.handle_key(_key("d"))
    editor.handle_key(_key("d"))
    assert editor.text == "a\nc"
    assert editor.cursor_row == 1


def test_x_deletes_char() -> None:
    editor = _editor("hello")
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("x"))
    assert editor.text == "ello"


def test_u_undoes_insert() -> None:
    editor = _editor("hello")
    editor.handle_key(Key("x", char="x"))
    assert editor.text == "xhello"
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("u"))
    assert editor.text == "hello"


def test_enter_submits_in_normal_mode() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    submitted: list[str] = []
    app.on_pi_editor_submitted = lambda message: submitted.append(message.text)  # type: ignore[method-assign]
    editor = _editor("hello")
    editor.app = app
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("enter"))
    assert submitted == ["hello"]


def test_o_opens_line_below_and_enters_insert() -> None:
    editor = _editor("a\nc")
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("o"))
    assert editor.vim_mode == "insert"
    assert editor.text == "a\n\nc"
    assert editor.cursor_row == 1


def test_i_a_enter_insert_mode() -> None:
    editor = _editor("hello")
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("i"))
    assert editor.vim_mode == "insert"
    editor.handle_key(_key("escape"))
    editor.handle_key(_key("a"))
    assert editor.vim_mode == "insert"
    assert editor.cursor_col == 1
