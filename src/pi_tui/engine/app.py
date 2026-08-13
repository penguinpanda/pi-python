"""App 基类：事件循环、渲染管线、焦点、overlay 合成、快捷键分发。"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
import time
from typing import Any

from rich.style import Style

from ..keybindings import KeybindingsManager
from ..overlay.layout import OverlayRect
from ..overlay.manager import OverlayHooks, OverlayManager
from ..overlay.model import OverlayOptions
from ..terminal import parse_osc11_background
from .cells import Line, blank_line, line_to_ansi
from .layout import LayoutFrame, box_at, render_layout_frame
from .keys import Key, KeyEvent, KeyParser, normalize_key_name
from .overlay_widget import OverlayWidget
from .terminal import FakeTerminal, ScreenBuffer, Terminal
from .widgets import Container, Message, ScrollView, Widget


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class App:
    """Textual-free 应用基类。"""

    def __init__(
        self,
        *,
        terminal: Terminal | FakeTerminal | None = None,
        size: tuple[int, int] = (80, 24),
        keybindings: KeybindingsManager | None = None,
        ui_mode: str = "regular",
    ) -> None:
        if terminal is None:
            try:
                terminal = Terminal(size=size)
            except Exception:
                terminal = FakeTerminal(size=size)
        self.terminal = terminal
        self.keybindings = keybindings or KeybindingsManager()
        self.ui_mode = ui_mode if ui_mode != "fullscreen" else "fullscreen"
        self._running = False
        self._events: asyncio.Queue[KeyEvent | None] = asyncio.Queue()
        self._render_requested = asyncio.Event()
        self._parser = KeyParser()
        self._buffer = ScreenBuffer(size[0], size[1])
        self.screen = Container(direction="vertical")
        self.screen.app = self
        self.focused: Widget | None = None
        self._tasks: set[asyncio.Task] = set()
        self.title = ""
        self._drag_target: Widget | None = None
        self._mouse_press_target: Widget | None = None
        self._mouse_select_start: tuple[int, int] | None = None
        self._mouse_select_current: tuple[int, int] | None = None
        self._mouse_selecting = False
        self._mouse_button_down = False
        self._mouse_word_select: tuple[int, int, int] | None = None
        self._mouse_last_press: tuple[float, tuple[int, int]] | None = None
        self._selection_autoscroll_direction = 0
        self._selection_autoscroll_task: Any = None
        self._selection_scroll_widget: Any = None
        self._scrollbar_hover: Any = None
        self._last_frame_lines: list[Line] = []
        self.osc_background: tuple[int, int, int] | None = None
        self.color_scheme: str | None = None
        self._kitty_protocol_active = False
        self.terminal_cell_size: tuple[int, int] | None = None
        self.open_url: Any | None = None
        self.clear_on_shrink = os.environ.get("PI_CLEAR_ON_SHRINK", "") in ("1", "true", "yes")
        self._regular_prev_lines: list[Line] = []
        self._regular_cursor_row = -1
        self._regular_hardware_cursor_row = -1
        self._regular_viewport_top = 0
        self._regular_prev_width = 0
        self._regular_prev_height = 0
        self._regular_max_lines_rendered = 0
        self._layout_frame: LayoutFrame | None = None
        self._overlays: list[OverlayWidget] = []
        self._overlay_manager = OverlayManager(
            OverlayHooks(
                make_widget=lambda key, lines, options: OverlayWidget(key, lines, options),
                update_widget=lambda widget, lines, options: self._update_overlay_widget(
                    widget, lines, options
                ),
                make_component_widget=lambda key, component, options: OverlayWidget(
                    key, [], options, component=component
                ),
                update_component=lambda widget, component, options: self._update_overlay_component(
                    widget, component, options
                ),
                mount=self._mount_overlay,
                remove=lambda widget: self._remove_overlay(widget),
                set_visible=lambda widget, visible: self._set_overlay_visible(widget, visible),
                reposition=self._reposition_overlay,
                focus=lambda widget: self.focus(widget),
                current_focus=lambda: self.focused,
                content_size=lambda widget: widget.content_size(),
                bring_to_front=self._bring_overlay_to_front,
                request_render=self.request_render,
            )
        )

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def request_render(self) -> None:
        if self._running:
            self._render_requested.set()

    def _compose(self) -> list[Line]:
        width, height = self.terminal.size
        frame = render_layout_frame(self.screen, width, height, self.request_render)
        self._layout_frame = frame
        lines = frame.lines
        for overlay in self._overlays:
            if not overlay.visible:
                continue
            row, col, overlay_width, overlay_height = overlay.rect
            if overlay_width <= 0 or overlay_height <= 0:
                continue
            overlay_lines = overlay.render(overlay_width, overlay_height)
            for index, line in enumerate(overlay_lines):
                target_row = row + index
                if 0 <= target_row < height:
                    target = lines[target_row]
                    if target.shared:
                        target = target.copy()
                        lines[target_row] = target
                    target.patch(col, line)
        if (
            self._mouse_selecting
            and self._mouse_select_start is not None
            and self._mouse_select_current is not None
        ):
            self._apply_selection_highlight(lines)
        return lines

    async def _render_if_requested(self, *, force: bool = False) -> None:
        if not force and not self._render_requested.is_set():
            return
        self._render_requested.clear()
        if self.ui_mode != "fullscreen":
            self._render_regular(force)
            self._write_hardware_cursor()
            self.terminal.flush()
            return
        size = self.terminal.size
        self._buffer.resize(size[0], size[1])
        lines = self._compose()
        self._last_frame_lines = lines
        command = self._buffer.diff(lines)
        if command:
            if getattr(self.terminal, "sync_output", False):
                command = "\x1b[?2026h" + command + "\x1b[?2026l"
            self.terminal.write(command)
        self._write_hardware_cursor()
        self.terminal.flush()

    def _compose_document(self) -> list[Line]:
        """regular 模式文档：布局引擎按自然高度合成 + overlay（对齐 TS TuiMainScreen）。"""
        width, _height = self.terminal.size
        frame = render_layout_frame(self.screen, width, None, self.request_render)
        self._layout_frame = frame
        lines = frame.lines
        for overlay in self._overlays:
            if not overlay.visible:
                continue
            row, col, overlay_width, overlay_height = overlay.rect
            if overlay_width <= 0 or overlay_height <= 0:
                continue
            overlay_lines = overlay.render(overlay_width, overlay_height)
            # 主屏文档比视口短时仍要容纳 overlay（对话框/选择器），
            # 对齐 TS TuiMainScreen.compositeOverlays 的 padding 行为。
            needed = row + overlay_height
            base_style = getattr(self.screen, "base_style", None)
            while len(lines) < needed:
                lines.append(blank_line(width, base_style))
            for index, line in enumerate(overlay_lines):
                target_row = row + index
                if 0 <= target_row < len(lines):
                    target = lines[target_row]
                    if target.shared:
                        target = target.copy()
                        lines[target_row] = target
                    target.patch(col, line)
        return lines

    def _render_regular(self, force: bool) -> None:
        """主屏模式渲染：差分写入 scrollback（对齐 TS TuiMainScreen）。

        只重写 firstChanged..lastChanged 变化行，光标用增量 A/B 移动；
        宽度/高度变化或内容收缩时整屏重写。
        """
        width, height = self.terminal.size
        lines = self._compose_document()
        self._last_frame_lines = lines
        prev = self._regular_prev_lines
        width_changed = self._regular_prev_width != 0 and self._regular_prev_width != width
        height_changed = self._regular_prev_height != 0 and self._regular_prev_height != height

        def full_render(clear: bool, clear_viewport: bool = False) -> None:
            buffer = "\x1b[?2026h"
            if clear:
                buffer += "\x1b[2J\x1b[H\x1b[3J"
            elif clear_viewport:
                # 首帧清视口但保留 scrollback，让文档（含 header）从视口顶部开始。
                buffer += "\x1b[2J\x1b[H"
            buffer += "\r\n".join(line_to_ansi(line, width) for line in lines)
            buffer += "\x1b[?2026l"
            self.terminal.write(buffer)
            self.terminal.flush()
            self._regular_prev_lines = list(lines)
            self._regular_prev_width = width
            self._regular_prev_height = height
            self._regular_cursor_row = max(0, len(lines) - 1)
            self._regular_hardware_cursor_row = self._regular_cursor_row
            self._regular_max_lines_rendered = len(lines)
            self._regular_viewport_top = max(0, max(height, len(lines)) - height)

        if force:
            self._regular_prev_lines = []
            self._regular_prev_width = 0
            self._regular_prev_height = 0

        # 首帧：清视口后从顶部输出（保留 scrollback，header 立即可见）。
        if not self._regular_prev_lines and not width_changed and not height_changed:
            full_render(False, clear_viewport=True)
            return
        # 宽度变化：换行方式变化，必须整屏重写。
        if width_changed:
            full_render(True)
            return
        # 高度变化：视口对齐需要整屏重写。
        if height_changed:
            full_render(True)
            return
        # 内容收缩：清掉残留行（可配置 PI_CLEAR_ON_SHRINK=0 关闭）。
        if self.clear_on_shrink and len(lines) < self._regular_max_lines_rendered:
            full_render(True)
            return

        prev_buffer_length = self._regular_prev_height if self._regular_prev_height > 0 else height
        prev_viewport_top = (
            max(0, prev_buffer_length - height) if height_changed else self._regular_viewport_top
        )
        viewport_top = prev_viewport_top
        hardware_cursor_row = self._regular_hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            target_screen_row = target_row - viewport_top
            return target_screen_row - current_screen_row

        first = -1
        last = -1
        for index in range(max(len(lines), len(prev))):
            old = prev[index] if index < len(prev) else None
            new = lines[index] if index < len(lines) else None
            # 跨帧复用同一 Line 对象（缓存行）时直接跳过；
            # 内容相同的重建行也无需重写。
            if old is new:
                continue
            if old is None or new is None or old != new:
                if first == -1:
                    first = index
                last = index
        appended = len(lines) > len(prev)
        if appended:
            if first == -1:
                first = len(prev)
            last = len(lines) - 1
        if first == -1:
            self._regular_viewport_top = prev_viewport_top
            self._regular_prev_height = height
            return

        # 变化全部在删除行：移动到新内容末尾，清掉多余行。
        if first >= len(lines):
            if len(prev) > len(lines):
                target_row = max(0, len(lines) - 1)
                if target_row < prev_viewport_top:
                    full_render(True)
                    return
                extra = len(prev) - len(lines)
                if extra > height:
                    full_render(True)
                    return
                buffer = "\x1b[?2026h"
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    buffer += f"\x1b[{line_diff}B"
                elif line_diff < 0:
                    buffer += f"\x1b[{-line_diff}A"
                buffer += "\r"
                clear_start_offset = 0 if len(lines) == 0 else 1
                if extra > 0 and clear_start_offset > 0:
                    buffer += f"\x1b[{clear_start_offset}B"
                for index in range(extra):
                    buffer += "\r\x1b[2K"
                    if index < extra - 1:
                        buffer += "\x1b[1B"
                move_back = max(0, extra - 1 + clear_start_offset)
                if move_back > 0:
                    buffer += f"\x1b[{move_back}A"
                buffer += "\x1b[?2026l"
                self.terminal.write(buffer)
                self.terminal.flush()
                self._regular_cursor_row = target_row
                self._regular_hardware_cursor_row = target_row
            self._regular_prev_lines = list(lines)
            self._regular_prev_width = width
            self._regular_prev_height = height
            self._regular_viewport_top = prev_viewport_top
            return

        # 变化区滚出视口：全量重写。
        if first < prev_viewport_top:
            full_render(True)
            return

        append_start = appended and first == len(prev) and first > 0
        buffer = "\x1b[?2026h"
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first - 1 if append_start else first
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                buffer += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buffer += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"
        buffer += "\r\n" if append_start else "\r"

        render_end = min(last, len(lines) - 1)
        for index in range(first, render_end + 1):
            if index > first:
                buffer += "\r\n"
            buffer += "\x1b[2K" + line_to_ansi(lines[index], width)
        final_cursor_row = render_end

        if len(prev) > len(lines):
            if render_end < len(lines) - 1:
                move_down = len(lines) - 1 - render_end
                buffer += f"\x1b[{move_down}B"
                final_cursor_row = len(lines) - 1
            extra = len(prev) - len(lines)
            for _ in range(extra):
                buffer += "\r\n\x1b[2K"
            buffer += f"\x1b[{extra}A"
        buffer += "\x1b[?2026l"

        self.terminal.write(buffer)
        self.terminal.flush()

        self._regular_cursor_row = max(0, len(lines) - 1)
        self._regular_hardware_cursor_row = final_cursor_row
        self._regular_max_lines_rendered = max(self._regular_max_lines_rendered, len(lines))
        self._regular_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)
        self._regular_prev_lines = list(lines)
        self._regular_prev_width = width
        self._regular_prev_height = height

    def _write_hardware_cursor(self) -> None:
        """把硬件光标定位到聚焦组件光标处（IME 候选窗口跟随输入光标）。

        对齐 TS：编辑器在光标格发射 CURSOR_MARKER，TUI 提取后移动硬件光标并
        显示；Python 用 widget.cursor_position() + 布局帧屏幕原点实现同等效果。
        PI_HARDWARE_CURSOR=0 可关闭。
        """
        if not self._hardware_cursor_enabled():
            return
        widget = self.focused
        if widget is None:
            self.terminal.hide_cursor()
            return
        local = widget.cursor_position()
        if local is None or self._layout_frame is None:
            self.terminal.hide_cursor()
            return
        row, col = _widget_screen_origin(self._layout_frame, widget)
        document_row = row + local[0]
        document_col = col + local[1]
        if self.ui_mode != "fullscreen":
            _width, height = self.terminal.size
            if not (
                self._regular_viewport_top <= document_row < self._regular_viewport_top + height
            ):
                self.terminal.hide_cursor()
                return
            # 差分渲染从当前硬件光标行继续移动，必须同步跟踪行号。
            self._regular_hardware_cursor_row = document_row
            screen_row = document_row - self._regular_viewport_top
        else:
            screen_row = document_row
        self.terminal.show_cursor()
        self.terminal.set_hardware_cursor(screen_row + 1, document_col + 1)

    def _hardware_cursor_enabled(self) -> bool:
        """默认开启（对齐 TS）；PI_HARDWARE_CURSOR=0/false/no 关闭。"""
        value = os.environ.get("PI_HARDWARE_CURSOR")
        if value is None:
            return True
        return value.strip().lower() not in ("", "0", "false", "no")

    # ------------------------------------------------------------------
    # 焦点 / 事件
    # ------------------------------------------------------------------

    def focus(self, widget: Widget | None) -> None:
        if self.focused is not None and self.focused is not widget:
            self.focused.focused = False
            self.focused.on_blur()
        self.focused = widget
        if widget is not None:
            widget.focused = True
            widget.on_focus()
        self.request_render()

    def dispatch_message(self, message: Message, namespace: str = "") -> None:
        """按命名空间解析处理器：on_<ns>_<snake> 或 on_<snake>。"""
        snake = _snake(type(message).__name__)
        handler: Any = None
        if namespace:
            handler = getattr(self, f"on_{namespace}_{snake}", None)
        if handler is None:
            handler = getattr(self, f"on_{snake}", None)
        if handler is not None:
            result = handler(message)
            if asyncio.iscoroutine(result):
                self._run_task(result)

    async def _handle_event(self, event: KeyEvent) -> None:
        if event.type == "key" and event.key is not None:
            self._handle_key(event.key)
        elif event.type == "paste":
            self._handle_paste(event.text)
        elif event.type == "mouse" and event.mouse is not None:
            self._handle_mouse(event.mouse)
        elif event.type == "resize":
            self._buffer.resize(event.width, event.height)
            self._overlay_manager.on_resize((event.width, event.height))
        elif event.type == "osc":
            self._handle_osc(event.data)
        elif event.type == "focus":
            if event.data == "out":
                # 失焦：取消进行中的拖选/滚动条操作（对齐 TS FOCUS_OUT）。
                self._clear_mouse_selection()
                self._set_scrollbar_hover(None)
        elif event.type == "kitty_flags":
            self._kitty_protocol_active = True
        elif event.type == "color_scheme":
            self.color_scheme = "light" if event.data == "2" else "dark"
        self.request_render()

    def _handle_osc(self, data: str) -> None:
        """处理运行时 OSC 响应：OSC 11 背景色通知。"""
        parsed = parse_osc11_background(f"\x1b]{data}\x07")
        if parsed is not None:
            self.osc_background = parsed

    def _handle_key(self, key: Key) -> None:
        # 对齐 TS：release 事件只分发给声明 wantsKeyRelease 的组件。
        if key.release and not getattr(self.focused, "wants_key_release", False):
            return
        # ctrl+c：存在鼠标选区时优先复制选区（编辑器自身选区仍由编辑器处理）。
        if (
            key.name == "ctrl+c"
            and self._mouse_selecting
            and self._mouse_select_start is not None
            and self._mouse_select_current is not None
        ):
            text = self._extract_selection(self._mouse_select_start, self._mouse_select_current)
            if text:
                self.copy_to_clipboard(text)
            self.request_render()
            return
        self._overlay_manager.route_input()
        if self.focused is not None and self.focused.handle_key(key):
            self.request_render()
            return
        if self._overlay_manager.handle_event(key):
            self.request_render()
            return
        self._dispatch_binding(key)

    def _dispatch_binding(self, key: Key) -> None:
        name = key.name
        action_id = self.keybindings.resolve(name)
        if action_id is None:
            normalized = normalize_key_name(name)
            if normalized != name:
                action_id = self.keybindings.resolve(normalized)
        if action_id is None:
            return
        binding = next(
            (b for b in self.keybindings.all_bindings() if b.action_id == action_id),
            None,
        )
        if binding is None:
            return
        action = getattr(self, f"action_{binding.action}", None)
        if action is None:
            return
        result = action()
        if asyncio.iscoroutine(result):
            self._run_task(result)

    def _handle_paste(self, text: str) -> None:
        widget = self.focused
        if isinstance(widget, Widget) and hasattr(widget, "insert"):
            widget.insert(text)
        elif isinstance(widget, Widget):
            for char in text:
                if not widget.handle_key(Key(name=char, char=char)):
                    break

    def _handle_mouse(self, event: Any) -> None:
        self._overlay_manager.route_input()
        event_type = getattr(event, "type", "")
        if event_type == "press":
            now = time.monotonic()
            cell = (event.row, event.col)
            double = (
                self._mouse_last_press is not None
                and now - self._mouse_last_press[0] < 0.4
                and self._mouse_last_press[1] == cell
            )
            self._mouse_last_press = (now, cell)
            self._mouse_button_down = True
            self._mouse_press_target = self._widget_at(event.row, event.col)
            self._selection_scroll_widget = (
                self._mouse_press_target
                if isinstance(self._mouse_press_target, ScrollView)
                else None
            )
            self._mouse_select_start = (event.row, event.col)
            self._mouse_select_current = (event.row, event.col)
            self._mouse_selecting = False
            self._mouse_word_select = None
            if double:
                word = self._word_at(event.row, event.col)
                if word is not None:
                    self._mouse_select_start = (event.row, word[0])
                    self._mouse_select_current = (event.row, word[1])
                    self._mouse_selecting = True
                    self._mouse_word_select = (event.row, word[0], word[1])
            target = self._drag_target or self._mouse_press_target
            if target is not None and target.handle_mouse(event):
                return
            return
        if event_type == "motion":
            if self._mouse_word_select is not None:
                return
            if self._mouse_select_start is not None and self._mouse_button_down:
                self._mouse_select_current = (event.row, event.col)
                if (event.row, event.col) != self._mouse_select_start:
                    self._mouse_selecting = True
                self._update_selection_autoscroll(event.row)
            else:
                self._update_scrollbar_hover(event.row, event.col)
            if self._drag_target is not None and self._drag_target.handle_mouse(event):
                return
            target = self._widget_at(event.row, event.col)
            if target is not None and target.handle_mouse(event):
                return
            return
        if event_type == "release":
            self._mouse_button_down = False
            if (
                self._mouse_selecting
                and self._mouse_select_start is not None
                and self._mouse_select_current is not None
            ):
                if self._mouse_word_select is not None:
                    self._mouse_select_start = (
                        self._mouse_word_select[0],
                        self._mouse_word_select[1],
                    )
                    self._mouse_select_current = (
                        self._mouse_word_select[0],
                        self._mouse_word_select[2],
                    )
                text = self._extract_selection(self._mouse_select_start, self._mouse_select_current)
                # 复制后保留选区高亮（对齐 TS：直到下次 press / 失焦才清除）。
                if text:
                    self.copy_to_clipboard(text)
                return
            # 单击 OSC8 链接：未拖动时打开 URL（对齐 TS openUrl）。
            if self._mouse_select_start is not None:
                link = self._link_at(*self._mouse_select_start)
                if link and self.open_url is not None:
                    self._clear_mouse_selection()
                    self.open_url(link)
                    return
            target = self._drag_target or self._mouse_press_target
            if target is not None:
                target.handle_mouse(event)
            self._clear_mouse_selection()
            return
        target = self._widget_at(event.row, event.col)
        if target is not None and target.handle_mouse(event):
            return
        if self.focused is not None and self.focused.handle_mouse(event):
            return

    def _extract_selection(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> str:
        """从最近一帧合成行提取拖选文本。"""
        row1, col1 = start
        row2, col2 = end
        if (row2, col2) < (row1, col1):
            row1, col1, row2, col2 = row2, col2, row1, col1
        lines = self._last_frame_lines or []
        if not lines:
            return ""
        row1 = max(0, min(row1, len(lines) - 1))
        row2 = max(0, min(row2, len(lines) - 1))
        col1 = max(0, col1)
        col2 = max(0, col2)
        parts: list[str] = []
        for row in range(row1, row2 + 1):
            text = lines[row].text()
            if row == row1 and row == row2:
                parts.append(text[col1 : col2 + 1])
            elif row == row1:
                parts.append(text[col1:])
            elif row == row2:
                parts.append(text[: col2 + 1])
            else:
                parts.append(text)
        return "\n".join(parts).rstrip()

    def _clear_mouse_selection(self) -> None:
        self._mouse_button_down = False
        if self._selection_autoscroll_task is not None:
            task = self._selection_autoscroll_task
            self._selection_autoscroll_task = None
            task.cancel()
        self._selection_autoscroll_direction = 0
        self._selection_scroll_widget = None
        self._mouse_press_target = None
        self._mouse_select_start = None
        self._mouse_select_current = None
        self._mouse_selecting = False
        self._mouse_word_select = None
        self._drag_target = None

    def _update_scrollbar_hover(self, row: int, col: int) -> None:
        """未按下时跟踪滚动条 hover（对齐 TS updateScrollbarHover）。"""
        target = self._widget_at(row, col)
        if isinstance(target, ScrollView):
            _r0, _c0, w, _h = target.rect
            if w > 2 and col == _c0 + w - 1:
                if target is not self._scrollbar_hover:
                    self._set_scrollbar_hover(target)
                return
        if self._scrollbar_hover is not None:
            self._set_scrollbar_hover(None)

    def _set_scrollbar_hover(self, widget: Any) -> None:
        if self._scrollbar_hover is not None and hasattr(
            self._scrollbar_hover, "set_scrollbar_active"
        ):
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = widget
        if widget is not None and hasattr(widget, "set_scrollbar_active"):
            widget.set_scrollbar_active(True)
        self.request_render()

    def _update_selection_autoscroll(self, row: int) -> None:
        """拖到视口边缘时启动/停止自动滚动（对齐 TS autoScrollSelection）。"""
        _width, height = self.terminal.size
        direction = 0
        if row <= 1:
            direction = -1
        elif row >= height - 2:
            direction = 1
        self._selection_autoscroll_direction = direction
        if direction != 0 and self._selection_autoscroll_task is None:
            self._selection_autoscroll_task = self._run_task(self._selection_autoscroll_loop())
        elif direction == 0 and self._selection_autoscroll_task is not None:
            task = self._selection_autoscroll_task
            self._selection_autoscroll_task = None
            task.cancel()

    async def _selection_autoscroll_loop(self) -> None:
        """每 50ms 滚动一次选区所在的 ScrollView。"""
        while True:
            await asyncio.sleep(0.05)
            direction = self._selection_autoscroll_direction
            if direction == 0:
                return
            widget = self._selection_scroll_widget
            if widget is not None and hasattr(widget, "scroll_offset"):
                widget.scroll_offset = max(0, widget.scroll_offset + direction)
                widget.refresh()
            self.request_render()

    def _apply_selection_highlight(self, lines: list[Line]) -> None:
        """拖动选区在合成帧上反色高亮。"""
        assert self._mouse_select_start is not None
        assert self._mouse_select_current is not None
        row1, col1 = self._mouse_select_start
        row2, col2 = self._mouse_select_current
        if (row2, col2) < (row1, col1):
            row1, col1, row2, col2 = row2, col2, row1, col1
        # 多行选区每帧逐格重建反色 Style 开销很大（Rich Style 加法昂贵）；
        # 按基础样式缓存反色结果，同一帧内相同样式只构建一次。
        reversed_cache: dict[Style | None, Style] = {}
        for row in range(row1, row2 + 1):
            if not (0 <= row < len(lines)):
                continue
            line = lines[row]
            if line.shared:
                line = line.copy()
                lines[row] = line
            start_col = col1 if row == row1 else 0
            end_col = col2 if row == row2 else len(line.cells) - 1
            for col in range(max(0, start_col), min(len(line.cells), end_col + 1)):
                cell = line.cells[col]
                base = cell.style
                reversed_style = reversed_cache.get(base)
                if reversed_style is None:
                    reversed_style = (base or Style()) + Style(reverse=True)
                    reversed_cache[base] = reversed_style
                cell.style = reversed_style

    def _link_at(self, row: int, col: int) -> str | None:
        """合成帧上指定单元格的 OSC8 链接（无则 None）。"""
        lines = self._last_frame_lines or []
        if not (0 <= row < len(lines)):
            return None
        cells = lines[row].cells
        if not (0 <= col < len(cells)):
            return None
        link = cells[col].link
        return link if isinstance(link, str) and link else None

    def _word_at(self, row: int, col: int) -> tuple[int, int] | None:
        """合成帧上双击位置的词边界 [start, end)。"""
        lines = self._last_frame_lines or []
        if not (0 <= row < len(lines)):
            return None
        text = lines[row].text()
        if not text:
            return None
        col = max(0, min(col, len(text) - 1))
        if not (text[col].isalnum() or text[col] == "_"):
            return None
        start = col
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        end = col
        while end + 1 < len(text) and (text[end + 1].isalnum() or text[end + 1] == "_"):
            end += 1
        return (start, end + 1)

    def _widget_at(self, row: int, col: int) -> Widget | None:
        """屏幕坐标 → 最上层可见组件（含 overlay；普通组件走 LayoutBox 树）。"""
        for overlay in reversed(self._overlays):
            if not overlay.visible:
                continue
            r0, c0, w, h = overlay.rect
            if r0 <= row < r0 + h and c0 <= col < c0 + w:
                return overlay
        if self._layout_frame is not None:
            box = box_at(self._layout_frame, col, row)
            if box is not None and isinstance(box.component, Widget):
                return box.component
        found: Widget | None = None
        for widget in self.screen.walk():
            if not widget.visible:
                continue
            r0, c0, w, h = widget.rect
            if w and h and r0 <= row < r0 + h and c0 <= col < c0 + w:
                found = widget
        return found

    # ------------------------------------------------------------------
    # Overlay hooks
    # ------------------------------------------------------------------

    def _mount_overlay(self, widget: OverlayWidget) -> None:
        widget.app = self
        widget.parent = self.screen
        self._overlays.append(widget)
        self.request_render()

    def _remove_overlay(self, widget: OverlayWidget) -> None:
        if widget in self._overlays:
            self._overlays.remove(widget)
        widget.app = None
        widget.parent = None
        self.request_render()

    def _set_overlay_visible(self, widget: OverlayWidget, visible: bool) -> None:
        widget.visible = visible
        self.request_render()

    def _reposition_overlay(
        self,
        widget: OverlayWidget,
        rect: OverlayRect,
        options: OverlayOptions,
    ) -> None:
        target = (rect.row, rect.col, rect.width, rect.height)
        if options.behavior.animate and widget.rect != target:
            existing = getattr(widget, "_anim_task", None)
            if existing is not None and not existing.done():
                existing.cancel()
            start = widget.rect
            duration = max(0.0, float(options.behavior.duration or 0.3))
            widget._anim_task = self._run_task(
                self._animate_overlay(widget, start, target, duration)
            )
            return
        widget.rect = target
        self.request_render()

    async def _animate_overlay(
        self,
        widget: OverlayWidget,
        start: tuple[int, int, int, int],
        target: tuple[int, int, int, int],
        duration: float,
    ) -> None:
        """out_cubic 插值动画 overlay 矩形。"""
        steps = max(1, int(duration * 30))
        for index in range(1, steps + 1):
            t = index / steps
            eased = 1 - (1 - t) ** 3
            row = round(start[0] + (target[0] - start[0]) * eased)
            col = round(start[1] + (target[1] - start[1]) * eased)
            width = round(start[2] + (target[2] - start[2]) * eased)
            height = round(start[3] + (target[3] - start[3]) * eased)
            widget.rect = (row, col, width, height)
            self.request_render()
            await asyncio.sleep(duration / steps)
        widget.rect = target
        self.request_render()

    def _bring_overlay_to_front(self, widget: OverlayWidget) -> None:
        if widget in self._overlays:
            self._overlays.remove(widget)
            self._overlays.append(widget)
        self.request_render()

    def _update_overlay_widget(
        self,
        widget: OverlayWidget,
        lines: list[str],
        options: OverlayOptions,
    ) -> None:
        widget.update_options(options)
        widget.update_content(lines)

    def _update_overlay_component(
        self,
        widget: OverlayWidget,
        component: Any,
        options: OverlayOptions,
    ) -> None:
        widget.update_options(options)
        widget.set_component(component)

    # ------------------------------------------------------------------
    # 剪贴板 / 任务 / 闪烁提示
    # ------------------------------------------------------------------

    def copy_to_clipboard(self, text: str) -> None:
        if self.terminal.copy_to_clipboard(text):
            return
        _platform_clipboard(text)

    def set_title(self, title: str) -> None:
        self.title = title
        if title:
            self.terminal.write(f"\x1b]2;{title}\x07")

    def flash(self, message: str, duration: float = 1.2) -> None:
        """全屏闪烁提示（对齐 TS AltScreenFlash）：居中非捕获 overlay。"""
        key = f"flash-{id(message):x}"
        self._overlay_manager.show(
            key,
            [message],
            {"nonCapturing": True, "anchor": "center"},
        )
        self._overlay_manager.reposition(key)
        self._run_task(self._auto_hide_flash(key, duration))

    async def _auto_hide_flash(self, key: str, duration: float) -> None:
        await asyncio.sleep(max(0.0, duration))
        self._overlay_manager.remove(key)

    def _run_task(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._report_task_exception)
        return task

    def _report_task_exception(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"Unhandled TUI task exception: {exc!r}", file=sys.stderr)

    def set_clear_on_shrink(self, enabled: bool) -> None:
        """内容收缩时清理残留行（差分渲染已覆盖，选项保留 API 对齐）。"""
        self.clear_on_shrink = bool(enabled)

    def exit(self) -> None:
        self._running = False

    def _clear_main_screen(self) -> None:
        """退出后清空主屏视口，只留下 shell 提示符。

        fullscreen 在恢复主屏后清屏；regular 模式直接清掉会话文档。
        两种模式都避免退出后残留 TUI 的 header / status / footer。
        """
        self.terminal.write("\x1b[2J\x1b[H")
        self.terminal.flush()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        pass

    def on_unmount(self) -> None:
        pass

    def on_tick(self) -> None:
        pass

    # 终端 cell 尺寸（CSI 16t 查询响应，仅图像渲染使用；对齐 TS queryCellSize）。
    _CELL_SIZE_RESPONSE = re.compile(rb"\x1b\[6;(\d+);(\d+)t")

    def _query_cell_size(self) -> None:
        """支持图像时查询终端 cell 像素尺寸（CSI 16 t，响应 CSI 6;h;w t）。"""
        try:
            from ..terminal_image import detect_capabilities

            if not detect_capabilities():
                return
        except Exception:
            return
        try:
            self.terminal.write("\x1b[16t")
        except Exception:
            pass

    def _consume_cell_size_response(self, data: bytes) -> bytes | None:
        """消费 CSI 6;h;w t 响应并记录尺寸，不阻塞其它输入；返回剩余数据。"""
        match = self._CELL_SIZE_RESPONSE.search(data)
        if match is None:
            return None
        self.terminal_cell_size = (int(match.group(2)), int(match.group(1)))
        remaining = data[: match.start()] + data[match.end() :]
        return remaining or None

    async def _read_loop(self) -> None:
        while self._running:
            chunk = await self.terminal.read_chunk()
            if chunk is None:
                await self._events.put(None)
                return
            if not chunk:
                # 空块（select 超时）：短暂休眠，避免忙轮询。
                await asyncio.sleep(0.05)
                continue
            # 终端 cell 尺寸响应不当作按键输入。
            remaining = self._consume_cell_size_response(chunk)
            if remaining is not None:
                chunk = remaining
            if not chunk:
                continue
            for event in self._parser.feed(chunk):
                await self._events.put(event)
            if self._parser.buffer:
                await asyncio.sleep(0.02)
                for event in self._parser.feed(b"", final=True):
                    await self._events.put(event)

    async def run_async(self) -> None:
        """运行应用直到 exit() / EOF。"""
        loop = asyncio.get_running_loop()
        handled_signals: list[int] = []
        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        ):
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self.exit)
                handled_signals.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        await self.terminal.enter(alt_screen=(self.ui_mode == "fullscreen"))
        self._running = True
        self._buffer.resize(*self.terminal.size)
        self.on_mount()
        self.request_render()
        self._query_cell_size()
        read_task = asyncio.create_task(self._read_loop())
        try:
            while self._running:
                await self._render_if_requested()
                resize_event = self.terminal.resize_event()
                if resize_event is not None:
                    await self._handle_event(resize_event)
                    continue
                try:
                    event = await asyncio.wait_for(self._events.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    self.on_tick()
                    continue
                if event is None:
                    break
                await self._handle_event(event)
                # 合并同一轮到达的输入（鼠标 motion 高频事件），批量处理后只渲染一次，
                # 避免每个事件都触发全量 diff/高亮导致拖选与复制卡顿。
                while True:
                    try:
                        queued = self._events.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if queued is None:
                        event = None
                        break
                    await self._handle_event(queued)
                if event is None:
                    break
            # 退出前只刷新待处理的增量帧；force=True 会整份重写文档，
            # 在 regular 模式把整个会话重复写进 scrollback。
            await self._render_if_requested()
        finally:
            self._running = False
            read_task.cancel()
            try:
                await read_task
            except (asyncio.CancelledError, Exception):
                pass
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self.on_unmount()
            await self.terminal.exit(alt_screen=(self.ui_mode == "fullscreen"))
            self._clear_main_screen()
            for sig in handled_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass


def _widget_screen_origin(frame: LayoutFrame | None, widget: Widget) -> tuple[int, int]:
    """组件在最近一帧中的屏幕坐标（scroll 内已平移），无帧时回退 widget.rect。"""
    if frame is not None:
        stack = [frame.root]
        while stack:
            box = stack.pop()
            if box.component is widget:
                return box.rect.y, box.rect.x
            stack.extend(box.children)
    row, col, _width, _height = widget.rect
    return row, col


def _platform_clipboard(text: str) -> None:
    """系统剪贴板回退（OSC 52 失败时）。"""
    try:
        if sys.platform == "win32":
            import subprocess

            subprocess.run(["clip"], input=text.encode("utf-16-le") + b"\x00\x00", check=False)
        elif sys.platform == "darwin":
            import subprocess

            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
        else:
            import subprocess

            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=False,
            )
    except Exception:
        pass


__all__ = ["App", "OverlayWidget"]
