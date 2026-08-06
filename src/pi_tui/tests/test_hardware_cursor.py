"""硬件光标 / IME 候选窗口定位测试（对齐 TS CURSOR_MARKER 行为）。"""

from __future__ import annotations

from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.widgets import Editor


def test_editor_cursor_position_uses_visible_columns() -> None:
    editor = Editor(text="你好ab")
    editor.cursor_col = 1  # 光标在“你”后：1 个宽字符 = 2 列
    assert editor.cursor_position() == (0, 2)
    editor.cursor_col = 2  # 光标在“你好”后：4 列
    assert editor.cursor_position() == (0, 4)
    editor.cursor_col = 3  # 光标在“你好a”后：5 列
    assert editor.cursor_position() == (0, 5)


async def test_hardware_cursor_follows_editor_caret_fullscreen() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term, ui_mode="fullscreen")
    editor = Editor(text="hello", border=True)
    app.screen.mount(editor, basis=3, grow=0, shrink=1, min_size=3)
    editor.focus()

    await app._render_if_requested(force=True)
    term.reset_output()

    editor.cursor_col = 2
    editor.refresh()
    await app._render_if_requested(force=True)

    # border 占 1 行 + cursor_row 0 → 屏幕行 2（1-based）；col 3（1-based）。
    assert "\x1b[2;3H" in term.output_text
    assert "\x1b[?25h" in term.output_text


async def test_hardware_cursor_follows_editor_caret_regular() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term, ui_mode="regular")
    editor = Editor(text="hello", border=True)
    app.screen.mount(editor, basis=3, grow=0, shrink=1, min_size=3)
    editor.focus()

    await app._render_if_requested(force=True)
    term.reset_output()

    editor.cursor_col = 2
    editor.refresh()
    await app._render_if_requested(force=True)

    assert "\x1b[2;3H" in term.output_text
    assert "\x1b[?25h" in term.output_text
    assert app._regular_hardware_cursor_row == 1


async def test_hardware_cursor_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PI_HARDWARE_CURSOR", "0")
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term, ui_mode="fullscreen")
    editor = Editor(text="hello", border=True)
    app.screen.mount(editor, basis=3, grow=0, shrink=1, min_size=3)
    editor.focus()

    await app._render_if_requested(force=True)
    term.reset_output()
    editor.cursor_col = 2
    editor.refresh()
    await app._render_if_requested(force=True)

    assert "\x1b[?25h" not in term.output_text
    assert "\x1b[2;3H" not in term.output_text
