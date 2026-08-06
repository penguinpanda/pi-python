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
from .cells import Line, line_to_ansi
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
        ui_mode: str = "fullscreen",
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
        self.open_url: Any | None = None
        self.clear_on_shrink = os.environ.get("PI_CLEAR_ON_SHRINK", "") in ("1", "true", "yes")
        self._regular_prev_lines: list[str] = []
        self._regular_cursor_row = -1
        self._regular_viewport_top = 0
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
        lines = self.screen.render(width, height)
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
                    lines[target_row].patch(col, line)
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
        """regular 模式文档：各组件自然高度 + overlay 合成（对齐 TS TuiMainScreen）。"""
        width, _height = self.terminal.size
        lines: list[Line] = []
        for widget in self.screen.children:
            if not widget.visible:
                continue
            if isinstance(widget, ScrollView) and widget.child is not None:
                _w, content_height = widget.child.content_size()
                lines.extend(widget.child.render(width, max(1, content_height)))
                continue
            spec = widget.height_spec
            if isinstance(spec, int):
                height = max(1, spec)
            else:
                height = max(1, widget.content_size()[1])
            lines.extend(widget.render(width, height))
        for overlay in self._overlays:
            if not overlay.visible:
                continue
            row, col, overlay_width, overlay_height = overlay.rect
            if overlay_width <= 0 or overlay_height <= 0:
                continue
            overlay_lines = overlay.render(overlay_width, overlay_height)
            for index, line in enumerate(overlay_lines):
                target_row = row + index
                if 0 <= target_row < len(lines):
                    lines[target_row].patch(col, line)
        return lines

    def _render_regular(self, force: bool) -> None:
        """主屏模式渲染：内容追加进 scrollback，必要时重写变化区。"""
        width, height = self.terminal.size
        lines = self._compose_document()
        self._last_frame_lines = lines
        ansi = [line_to_ansi(line, width) for line in lines]
        prev = self._regular_prev_lines
        first = 0
        while first < len(prev) and first < len(ansi) and prev[first] == ansi[first]:
            first += 1
        if first == len(prev) and first == len(ansi):
            return
        buffer = ""
        if first >= len(prev):
            # 追加：从文档末尾继续写。
            if prev:
                buffer += "\r\n"
            buffer += "\r\n".join(ansi[first:])
            self._regular_cursor_row = len(ansi) - 1
        else:
            viewport_top = max(0, self._regular_cursor_row - height + 1)
            if first < viewport_top:
                # 改动区已滚出视口：全量重写（对齐 TS fullRender）。
                buffer = "\x1b[2J\x1b[H" + "\r\n".join(ansi)
                self._regular_viewport_top = 0
                self._regular_cursor_row = len(ansi) - 1
            else:
                delta = first - self._regular_cursor_row
                if delta > 0:
                    buffer += f"\x1b[{delta}B"
                elif delta < 0:
                    buffer += f"\x1b[{-delta}A"
                for index in range(first, len(ansi)):
                    buffer += "\r\x1b[2K" + ansi[index]
                    if index < len(ansi) - 1:
                        buffer += "\r\n"
                if len(ansi) < len(prev):
                    extra = len(prev) - len(ansi)
                    for _ in range(extra):
                        buffer += "\r\n\x1b[2K"
                    if extra:
                        buffer += f"\x1b[{extra}A"
                self._regular_cursor_row = len(ansi) - 1
        self._regular_prev_lines = ansi
        self._regular_viewport_top = max(0, self._regular_cursor_row - height + 1)
        if buffer:
            if getattr(self.terminal, "sync_output", False):
                buffer = "\x1b[?2026h" + buffer + "\x1b[?2026l"
            self.terminal.write(buffer)
            self.terminal.flush()

    def _write_hardware_cursor(self) -> None:
        """PI_HARDWARE_CURSOR=1 时用硬件光标定位（编辑器提供位置）。"""
        if os.environ.get("PI_HARDWARE_CURSOR", "") not in ("1", "true", "yes"):
            return
        widget = self.focused
        if widget is None:
            return
        local = widget.cursor_position()
        if local is None:
            return
        row, col, _width, _height = widget.rect
        self.terminal.set_hardware_cursor(row + local[0] + 1, col + local[1] + 1)

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
        for row in range(row1, row2 + 1):
            if not (0 <= row < len(lines)):
                continue
            line = lines[row]
            start_col = col1 if row == row1 else 0
            end_col = col2 if row == row2 else len(line.cells) - 1
            for col in range(max(0, start_col), min(len(line.cells), end_col + 1)):
                cell = line.cells[col]
                cell.style = (cell.style or Style()) + Style(reverse=True)

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
        """屏幕坐标 → 最上层可见组件（含 overlay）。"""
        for overlay in reversed(self._overlays):
            if not overlay.visible:
                continue
            r0, c0, w, h = overlay.rect
            if r0 <= row < r0 + h and c0 <= col < c0 + w:
                return overlay
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
        return task

    def set_clear_on_shrink(self, enabled: bool) -> None:
        """内容收缩时清理残留行（差分渲染已覆盖，选项保留 API 对齐）。"""
        self.clear_on_shrink = bool(enabled)

    def exit(self) -> None:
        self._running = False

    def _write_main_screen_document(self) -> None:
        """退出 alt-screen 后把最后一帧文档写入主屏（对齐 TS 退出行为）。

        TS 在 stop 时切换到 regular renderer 渲染一帧再退出，主屏保留 TUI
        内容，光标落在内容之后的新行，shell 提示符紧接 footer。
        """
        if self.ui_mode != "fullscreen":
            return
        width, _height = self.terminal.size
        lines = self._compose_document()
        if not lines:
            return
        document = "\r\n".join(f"\r\x1b[2K{line_to_ansi(line, width)}" for line in lines)
        self.terminal.write(f"{document}\x1b[0m\r\n\x1b[?25h")
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
            await self._render_if_requested(force=True)
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
            self._write_main_screen_document()
            for sig in handled_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass


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
