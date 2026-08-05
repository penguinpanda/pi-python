"""PiEditorVim（vim 模式编辑器）测试。"""

from __future__ import annotations

import pytest
from textual.app import App

from pi_tui.components import PiEditorVim


class _VimHarness(App):
    def __init__(self, editor: PiEditorVim) -> None:
        super().__init__()
        self._editor = editor
        self.submitted: list[str] = []

    def compose(self):
        yield self._editor

    def on_pi_editor_submitted(self, event) -> None:
        self.submitted.append(event.text)


def _editor(text: str) -> PiEditorVim:
    editor = PiEditorVim()
    editor.text = text
    return editor


@pytest.mark.asyncio
async def test_escape_toggles_mode() -> None:
    editor = _editor("hello")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        assert editor.vim_mode == "insert"
        await pilot.press("escape")
        assert editor.vim_mode == "normal"
        await pilot.press("escape")
        assert editor.vim_mode == "insert"


@pytest.mark.asyncio
async def test_normal_mode_navigation_jk() -> None:
    editor = _editor("line1\nline2\nline3")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("j")
        assert editor.cursor_location[0] == 1
        await pilot.press("j")
        assert editor.cursor_location[0] == 2
        await pilot.press("k")
        assert editor.cursor_location[0] == 1


@pytest.mark.asyncio
async def test_dd_deletes_current_line() -> None:
    editor = _editor("a\nb\nc")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        editor.move_cursor((1, 0))
        await pilot.press("escape")
        await pilot.press("d", "d")
        assert editor.document.lines == ["a", "c"]


@pytest.mark.asyncio
async def test_x_deletes_char() -> None:
    editor = _editor("abc")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        editor.move_cursor((0, 1))
        await pilot.press("escape")
        await pilot.press("x")
        assert editor.text == "ac"


@pytest.mark.asyncio
async def test_u_undoes_insert() -> None:
    editor = _editor("abc")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        editor.move_cursor((0, 3))
        await pilot.press("z")
        assert editor.text == "abcz"
        await pilot.press("escape")
        await pilot.press("u")
        assert editor.text == "abc"


@pytest.mark.asyncio
async def test_enter_submits_in_normal_mode() -> None:
    editor = _editor("hi")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.pause()
        assert app.submitted == ["hi"]


@pytest.mark.asyncio
async def test_o_opens_line_below_and_enters_insert() -> None:
    editor = _editor("a")
    app = _VimHarness(editor)
    async with app.run_test() as pilot:
        editor.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("o")
        assert editor.document.lines == ["a", ""]
        assert editor.vim_mode == "insert"
