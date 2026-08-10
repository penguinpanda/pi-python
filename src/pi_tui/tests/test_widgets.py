"""滚动条拖拽 / AltScreenFlash / overlay 动画 / 鼠标选择 / 图像 passthrough 测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_tui.components import MessageEntry
from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.cells import Cell, Line, blank_line
from pi_tui.engine.keys import Key, KeyEvent, MouseEvent, parse_input
from pi_tui.engine.terminal import ScreenBuffer
from pi_tui.engine.widgets import (
    AltScreenFlash,
    Container,
    Editor,
    ScrollView,
    Static,
    Vertical,
    Widget,
)
from pi_tui.terminal_image import TerminalImage, encode_kitty_delete


def _mouse(type_: str, row: int, col: int, button: str = "left") -> MouseEvent:
    return MouseEvent(type=type_, button=button, row=row, col=col)


def test_sgr_mouse_coordinates_are_zero_based() -> None:
    events = parse_input(b"\x1b[<0;3;5M")
    mouse = events[0].mouse
    assert mouse is not None
    assert (mouse.row, mouse.col) == (4, 2)


def test_cjk_lines_never_exceed_terminal_width() -> None:
    from rich.cells import cell_len

    from pi_tui.engine.cells import line_from_text, line_to_ansi
    from pi_tui.engine.text import strip_ansi

    # 中文按 2 列计：不拆宽字符、不超宽。
    line = line_from_text("加内存", 6)
    assert len(line.cells) == 3
    assert cell_len(line.text()) == 6
    line = line_from_text("加内存", 4)
    assert "".join(c.char for c in line.cells).startswith("加内")
    assert cell_len(line.text()) == 4

    ansi = line_to_ansi(line_from_text("加内存" * 30, 20), 20)
    assert cell_len(strip_ansi(ansi)) == 20

    buffer = ScreenBuffer(10, 3)
    normalized = buffer._normalize([line_from_text("加内存" * 30, 10)])
    assert cell_len(normalized[0].text()) == 10


def test_editor_cursor_highlight_uses_character_cell_for_cjk() -> None:
    """回归：CJK 光标高亮应落在字符格而不是可见列，避免每次输入一个
    宽字符就多空一列。"""
    editor = Editor(text="你好", border=True)
    editor.focused = True
    editor.cursor_col = 2
    lines = editor.render(10, 3)
    content = lines[1]
    reverse_indices = [
        index
        for index, cell in enumerate(content.cells)
        if cell.style is not None and cell.style.reverse
    ]
    assert reverse_indices == [2]
    assert content.cells[2].char == " "


def test_empty_static_has_zero_natural_size() -> None:
    assert Static("").content_size() == (0, 0)
    assert Static("x").content_size()[1] == 1


def test_container_allocate_preserves_mount_order() -> None:
    """fixed / auto / fr 子组件按挂载顺序排列，而不是按类型分组。"""
    container = Container(direction="vertical")
    header = Static("h", height=1)
    chat = Static("c", height="1fr")
    status = Static("s", height=1)
    widgets_above = Static("", height="auto")
    editor = Static("e", height=6)
    widgets_below = Static("", height="auto")
    footer = Static("f", height=1)
    for child in (header, chat, status, widgets_above, editor, widgets_below, footer):
        container.mount(child)

    lines = container.render(100, 24)

    assert header.rect == (0, 0, 100, 1)
    assert chat.rect == (1, 0, 100, 15)
    assert status.rect == (16, 0, 100, 1)
    assert widgets_above.rect[3] == 0
    assert editor.rect == (17, 0, 100, 6)
    assert widgets_below.rect[3] == 0
    assert footer.rect == (23, 0, 100, 1)
    assert len(lines) == 24


def test_scrollview_mount_propagates_app_to_child() -> None:
    class _StubApp:
        def request_render(self) -> None:
            pass

    app = _StubApp()
    body = Vertical()
    body.mount(Static("x", height=1))
    view = ScrollView(body)
    view.app = app
    view.mount_child(body)
    assert view.child is not None
    assert view.child.app is app
    assert view.child.children[0].app is app


def test_editor_history_up_down_recalls_prompts() -> None:
    editor = Editor()
    editor.add_to_history("first prompt")
    editor.add_to_history("second prompt")

    editor.handle_key(Key(name="up"))
    assert editor.text == "second prompt"
    editor.handle_key(Key(name="up"))
    assert editor.text == "first prompt"
    editor.handle_key(Key(name="down"))
    assert editor.text == "second prompt"
    editor.handle_key(Key(name="down"))
    assert editor.text == ""
    assert editor.history_index == -1

    # 输入会退出历史浏览，在已召回文本上继续编辑（对齐 TS）。
    editor.handle_key(Key(name="up"))
    editor.insert("draft")
    assert editor.history_index == -1
    assert editor.text == "draftsecond prompt"

    # 非空首行中间按上：跳到行首而不是翻历史。
    editor.text = "abc"
    editor.handle_key(Key(name="up"))
    assert editor.text == "abc"
    assert editor.cursor_col == 0


def test_editor_ts_emacs_and_alt_bindings() -> None:
    editor = Editor()
    editor.text = "one two three"

    editor.move_cursor((0, len("one two three")))
    editor.handle_key(Key(name="ctrl+b"))
    assert editor.cursor_col == len("one two three") - 1
    editor.handle_key(Key(name="ctrl+f"))
    assert editor.cursor_col == len("one two three")

    editor.handle_key(Key(name="alt+b"))
    assert editor.cursor_col == 8  # 'three' 词首
    editor.handle_key(Key(name="alt+f"))
    assert editor.cursor_col == len("one two three")

    editor.text = "one two three"
    editor.move_cursor((0, 7))
    editor.handle_key(Key(name="alt+backspace"))
    assert editor.text == "one  three"

    editor.text = "one two three"
    editor.move_cursor((0, 4))
    editor.handle_key(Key(name="alt+d"))
    assert editor.text == "one three"

    editor.undo()
    assert editor.text == "one two three"
    editor.handle_key(Key(name="ctrl+-"))
    assert editor.text == "one  three"

    editor.text = "abc abc abc"
    editor.move_cursor((0, 0))
    editor.handle_key(Key(name="ctrl+]"))
    editor.handle_key(Key(name="b", char="b"))
    assert editor.cursor_col == 1
    editor.handle_key(Key(name="ctrl+alt+]"))
    editor.handle_key(Key(name="b", char="b"))
    assert editor.cursor_col == 1


def test_scrollview_drag_updates_offset() -> None:
    body = Vertical()
    for index in range(20):
        body.mount(Static(f"line {index}", height=1))
    view = ScrollView(body)
    view.rect = (0, 0, 12, 5)
    assert view.handle_mouse(_mouse("press", 4, 11)) is True
    assert view._dragging is True
    view.handle_mouse(_mouse("motion", 2, 11))
    assert 0 <= view.scroll_offset <= 15
    assert view.handle_mouse(_mouse("release", 2, 11)) is True
    assert view._dragging is False


def test_scrollview_press_outside_scrollbar_ignored() -> None:
    view = ScrollView(Static("x", height=1))
    view.rect = (0, 0, 12, 5)
    assert view.handle_mouse(_mouse("press", 1, 2)) is False
    assert view._dragging is False


def test_alt_screen_flash_centers_text() -> None:
    flash = AltScreenFlash("Done")
    lines = flash.render(20, 5)
    assert "Done" in lines[2].text()
    assert lines[0].text().strip() == ""


@pytest.mark.asyncio
async def test_overlay_animation_reaches_target() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.1)
    app._overlay_manager.show(
        "anim",
        ["x"],
        {"anchor": "center", "animate": True, "duration": 0.2},
    )
    app._overlay_manager.reposition("anim")
    await asyncio.sleep(0.5)
    entry = app._overlay_manager.get("anim")
    assert entry is not None
    row, col, width, height = entry.widget.rect
    assert (width, height) == (3, 3)
    assert row >= 0 and col >= 0
    app.exit()
    await asyncio.sleep(0.1)
    await task


@pytest.mark.asyncio
async def test_app_drag_selection_copies_text() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    app.screen.mount(Static("hello world", height=1))
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("motion", 0, 4, button="none"))
    app._handle_mouse(_mouse("release", 0, 4, button="none"))
    assert term.clipboard == ["hello"]
    app.exit()
    await asyncio.sleep(0.1)
    await task


@pytest.mark.asyncio
async def test_double_click_selects_word() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    app.screen.mount(Static("hello world", height=1))
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    app._handle_mouse(_mouse("press", 0, 1))
    app._handle_mouse(_mouse("release", 0, 1, button="none"))
    app._handle_mouse(_mouse("press", 0, 1))
    app._handle_mouse(_mouse("release", 0, 1, button="none"))
    assert term.clipboard == ["hello"]
    app.exit()
    await asyncio.sleep(0.1)
    await task


@pytest.mark.asyncio
async def test_drag_selection_highlight() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    app.screen.mount(Static("hello world", height=1))
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("motion", 0, 4, button="none"))
    composed = app._compose()
    assert any(cell.style is not None and cell.style.reverse for cell in composed[0].cells[:5])
    app._handle_mouse(_mouse("release", 0, 4, button="none"))
    app.exit()
    await asyncio.sleep(0.1)
    await task


@pytest.mark.asyncio
async def test_message_click_copies_on_release() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    copied: list[str] = []
    app.on_copy_requested = lambda message: copied.append(message.text)  # type: ignore[method-assign]
    entry = MessageEntry("User", "click me")
    app.screen.mount(entry)
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("release", 0, 0, button="none"))
    assert copied == ["click me"]
    app.exit()
    await asyncio.sleep(0.1)
    await task


def test_markdown_body_path_linkified(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "wezterm")
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    entry = MessageEntry("User", f"see {target} now")
    lines = entry.render(80, 10)
    linked = [cell.link for line in lines for cell in line.cells if cell.link]
    assert linked and linked[0] == target.resolve().as_uri()


def test_message_entry_kitty_passthrough(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    entry = MessageEntry("Assistant", "hello", images=[b"abc"])
    lines = entry.render(40, 10)
    assert lines[1].passthrough.startswith("\x1b_Ga=p")


def test_message_entry_iterm2_passthrough(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    entry = MessageEntry("Assistant", "hello", images=[b"abc"])
    lines = entry.render(40, 10)
    assert lines[1].passthrough.startswith("\x1b]1337;File=name=image-0;inline=1:")


def test_message_entry_image_placeholder_without_capability(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    entry = MessageEntry("Assistant", "hello", images=[b"abc"])
    lines = entry.render(40, 10)
    assert lines[1].passthrough == ""


def test_message_entry_removal_emits_kitty_delete(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    entry = MessageEntry("Assistant", "hello", images=[b"abc"])
    app.screen.mount(entry)
    term.reset_output()
    entry.remove()
    assert any(encode_kitty_delete(entry._image_id_base) in part for part in term.output)


def test_terminal_image_render_passthrough(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    image = TerminalImage(b"abc", name="pic.png")
    lines = image.render(20, 1)
    assert lines[0].passthrough.startswith("\x1b_Ga=p")


def test_terminal_image_fallback_placeholder(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    image = TerminalImage(b"abc", name="pic.png")
    lines = image.render(20, 1)
    assert lines[0].passthrough == ""
    assert lines[0].text().startswith("[image: pic.png]")


def test_screen_buffer_emits_passthrough_only_on_change() -> None:
    buffer = ScreenBuffer(10, 3)
    line = Line([Cell("x") for _ in range(10)], passthrough="\x1b_Gseq")
    first = buffer.diff([line, blank_line(10), blank_line(10)])
    assert "\x1b_Gseq" in first
    second = buffer.diff([line, blank_line(10), blank_line(10)])
    assert "\x1b_Gseq" not in second


@pytest.mark.asyncio
async def test_mouse_click_opens_osc8_link() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    opened: list[str] = []
    app.open_url = opened.append
    cells = [Cell("x", link="https://example.com")] + [Cell(" ") for _ in range(79)]
    app._last_frame_lines = [Line(cells)]
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("release", 0, 0, button="none"))
    assert opened == ["https://example.com"]


@pytest.mark.asyncio
async def test_focus_out_clears_selection() -> None:
    app = App(terminal=FakeTerminal(size=(80, 24)))
    app._mouse_select_start = (0, 0)
    app._mouse_select_current = (0, 3)
    app._mouse_selecting = True
    await app._handle_event(KeyEvent(type="focus", data="out"))
    assert app._mouse_select_start is None
    assert app._mouse_selecting is False


def test_editor_border_and_padding_render() -> None:
    editor = Editor(border=True, padding_x=2)
    editor.text = "hi"
    lines = editor.render(10, 6)
    assert lines[0].text().strip() == "─" * 10
    assert lines[-1].text().strip() == "─" * 10
    assert lines[1].text().startswith("  hi")
    assert editor.cursor_position() == (1, 4)


def test_editor_without_border_unchanged() -> None:
    editor = Editor()
    editor.text = "hi"
    lines = editor.render(10, 4)
    assert lines[0].text().strip() == "hi"
    assert editor.cursor_position() == (0, 2)


@pytest.mark.asyncio
async def test_drag_selection_autoscroll_starts_and_stops() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    view = ScrollView(Static("x", height=1))
    view.rect = (0, 0, 80, 20)
    app._mouse_select_start = (0, 0)
    app._mouse_select_current = (0, 0)
    app._mouse_button_down = True
    app._selection_scroll_widget = view
    app._handle_mouse(_mouse("motion", 22, 5, button="none"))
    assert app._selection_autoscroll_direction == 1
    assert app._selection_autoscroll_task is not None
    app._handle_mouse(_mouse("motion", 10, 5, button="none"))
    assert app._selection_autoscroll_direction == 0
    assert app._selection_autoscroll_task is None
    app._clear_mouse_selection()


@pytest.mark.asyncio
async def test_selection_includes_overlay_text() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    app.screen.mount(Static("hello world", height=1))
    app._overlay_manager.show("ov", ["OVERLAY"], {"anchor": "center"})
    app._overlay_manager.reposition("ov")
    await app._render_if_requested(force=True)
    overlay = app._overlays[-1]
    row, col = overlay.rect[0], overlay.rect[1]
    app._handle_mouse(_mouse("press", row, col))
    app._handle_mouse(_mouse("motion", row, col + 7, button="none"))
    app._handle_mouse(_mouse("release", row, col + 7, button="none"))
    assert term.clipboard and "OVERLAY" in term.clipboard[0]


@pytest.mark.asyncio
async def test_regular_mode_renders_without_alt_screen() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term, ui_mode="regular")
    app.screen.mount(Static("hello world", height=1))
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.2)
    term.feed_text("x")
    await asyncio.sleep(0.1)
    app.exit()
    await asyncio.sleep(0.1)
    await task
    output = "".join(term.output)
    assert "\x1b[?1049h" not in output
    assert "\x1b[?1049l" not in output
    assert "hello world" in output


def test_color_scheme_notification_parses() -> None:
    events = parse_input(b"\x1b[?997;2n")
    assert [(event.type, event.data) for event in events] == [("color_scheme", "2")]


@pytest.mark.asyncio
async def test_color_scheme_event_sets_app_state() -> None:
    app = App(terminal=FakeTerminal(size=(80, 24)))
    await app._handle_event(KeyEvent(type="color_scheme", data="2"))
    assert app.color_scheme == "light"
    await app._handle_event(KeyEvent(type="color_scheme", data="1"))
    assert app.color_scheme == "dark"


@pytest.mark.asyncio
async def test_scrollbar_hover_highlights() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    view = ScrollView(Static("x", height=1))
    app.screen.mount(view)
    view.rect = (0, 0, 12, 5)
    app._handle_mouse(_mouse("motion", 2, 11, button="none"))
    assert view.scrollbar_active is True
    app._handle_mouse(_mouse("motion", 2, 5, button="none"))
    assert view.scrollbar_active is False


def test_scroll_by_returns_remaining() -> None:
    body = Vertical()
    for index in range(10):
        body.mount(Static(f"line {index}", height=1))
    view = ScrollView(body)
    view.rect = (0, 0, 12, 5)
    assert view.scroll_by(3) == 0
    assert view.scroll_offset == 3
    assert view.scroll_by(10) == 8  # 到顶只消费 2 行
    assert view.scroll_offset == 5


def test_markdown_heading_uses_theme_color() -> None:
    from pi_tui.engine.text import render_markdown

    from rich.color import Color

    lines = render_markdown("# Title", 40, theme_colors={"heading": (255, 0, 0)})
    expected = Color.from_rgb(255, 0, 0)
    assert any(
        cell.style is not None and cell.style.color == expected
        for line in lines
        for cell in line.cells
    )


def test_editor_ctrl_shift_home_end_selects_document() -> None:
    editor = Editor()
    editor.text = "first line\nsecond line\nthird line"
    editor.move_cursor((1, len("second line")))
    editor.handle_key(Key(name="ctrl+shift+home"))
    assert editor.selection == ((0, 0), (1, len("second line")))
    editor.move_cursor((1, 0))
    editor.handle_key(Key(name="ctrl+shift+end"))
    assert editor.selection == ((1, 0), (2, len("third line")))
    assert editor.selected_text == "second line\nthird line"


@pytest.mark.asyncio
async def test_release_key_filtered_unless_wanted() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)

    class _W(Widget):
        def __init__(self) -> None:
            super().__init__(focusable=True)
            self.wants_key_release = True
            self.calls: list[Key] = []

        def handle_key(self, key: Key) -> bool:
            self.calls.append(key)
            return True

    widget = _W()
    app.screen.mount(widget)
    app.focus(widget)
    await app._handle_event(KeyEvent(type="key", key=Key("a", char="a", release=True)))
    assert len(widget.calls) == 1
    widget.wants_key_release = False
    await app._handle_event(KeyEvent(type="key", key=Key("b", char="b", release=True)))
    assert len(widget.calls) == 1


@pytest.mark.asyncio
async def test_drag_release_copies_keeps_highlight_and_no_stuck_selection() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    cells = [Cell("a"), Cell("b"), Cell("c")] + [Cell(" ") for _ in range(77)]
    app._last_frame_lines = [Line(cells)]
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("motion", 0, 2, button="none"))
    app._handle_mouse(_mouse("release", 0, 2, button="none"))
    assert term.clipboard == ["abc"]
    # release 后保留高亮，但无按键移动不再扩展选区。
    assert app._mouse_selecting is True
    assert app._mouse_select_current == (0, 2)
    app._handle_mouse(_mouse("motion", 0, 5, button="none"))
    assert app._mouse_select_current == (0, 2)
    assert term.clipboard == ["abc"]


@pytest.mark.asyncio
async def test_ctrl_c_copies_mouse_selection() -> None:
    term = FakeTerminal(size=(80, 24))
    app = App(terminal=term)
    cells = [Cell("x"), Cell("y"), Cell("z")] + [Cell(" ") for _ in range(77)]
    app._last_frame_lines = [Line(cells)]
    app._handle_mouse(_mouse("press", 0, 0))
    app._handle_mouse(_mouse("motion", 0, 2, button="none"))
    await app._handle_event(KeyEvent(type="key", key=Key(name="ctrl+c")))
    assert term.clipboard == ["xyz"]


def test_editor_border_uses_dedicated_style() -> None:
    from rich.color import Color
    from rich.style import Style

    border = Style(color=Color.from_rgb(69, 71, 90), bgcolor=Color.from_rgb(30, 30, 46))
    editor = Editor(border=True, base_style=Style(color=Color.from_rgb(205, 214, 244)))
    editor.border_style = border
    editor.focused = True
    lines = editor.render(10, 4)
    assert lines[0].cells[0].style is not None
    assert lines[0].cells[0].style.color == Color.from_rgb(69, 71, 90)
    assert lines[-1].cells[0].style.color == Color.from_rgb(69, 71, 90)
    # 内容行首个单元格是光标（反色），不是边框色。
    assert lines[1].cells[0].style is not None
    assert lines[1].cells[0].style.reverse is True


def test_line_to_ansi_turns_off_reverse_and_keeps_rgb() -> None:
    from pi_tui.engine.cells import line_to_ansi
    from rich.color import Color
    from rich.style import Style

    base = Style(color=Color.from_rgb(166, 173, 200), bgcolor=Color.from_rgb(30, 30, 46))
    line = Line(
        [
            Cell(" ", base + Style(reverse=True)),
            Cell(" ", base),
            Cell(" ", base),
            Cell(" ", base),
        ]
    )
    ansi = line_to_ansi(line, 4)
    # 光标格：反色 + 完整 RGB；后续格补发 27 关闭反色，RGB 参数不被去重。
    assert "\x1b[7;38;2;166;173;200;48;2;30;30;46m" in ansi
    assert "\x1b[38;2;166;173;200;48;2;30;30;46;27m" in ansi
