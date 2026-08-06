"""组件树：Widget / Container / Text / Input / Editor / ScrollView / 列表。

对齐 TS packages/tui/src/components/：每个组件 render(width, height) 生成
定宽 Line 列表，容器按单元格合成，App 按行差分写入终端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from rich.style import Style

from .cells import Cell, Line, blank_line, line_from_text
from .keys import Key, MouseEvent
from .layout import render_layout_frame
from .layout_node import ScrollLayoutNode, StackLayoutEntry, StackLayoutNode
from .text import render_markdown, render_markup


class Message:
    """组件事件消息。"""

    def __init__(self) -> None:
        self.stop = False

    def stop_propagation(self) -> None:
        self.stop = True


def _style(base: Style | str | None) -> Style | None:
    if base is None:
        return None
    if isinstance(base, Style):
        return base
    return Style.parse(base)


def _natural_size_of(component: Any, width: int) -> tuple[int, int]:
    method = getattr(component, "natural_size", None)
    if callable(method):
        try:
            return method(width)
        except Exception:
            pass
    return component.content_size()


class Widget:
    """组件基类。"""

    def __init__(
        self,
        *,
        id: str | None = None,
        visible: bool = True,
        height: int | str = "auto",
        width: int | str = "1fr",
        base_style: Style | str | None = None,
        focusable: bool = False,
    ) -> None:
        self.id = id
        self.visible = visible
        self.height_spec = height
        self.width_spec = width
        self.base_style = _style(base_style)
        self.focusable = focusable
        self.parent: Widget | None = None
        self.children: list[Widget] = []
        self.app: Any = None
        self.focused = False
        self.rect = (0, 0, 0, 0)  # row, col, width, height（布局后）
        self.wants_key_release = False  # 对齐 TS wantsKeyRelease

    # ------------------------------------------------------------------
    # 树操作
    # ------------------------------------------------------------------

    def mount(self, child: "Widget") -> "Widget":
        child.parent = self
        self.children.append(child)
        for descendant in child.walk():
            descendant.app = self.app
        self.refresh()
        return child

    def remove(self, child: "Widget | None" = None) -> None:
        if child is None:
            if self.parent is not None:
                self.parent.remove(self)
            return
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            for descendant in child.walk():
                descendant.app = None
            self.refresh()

    def refresh(self) -> None:
        if self.app is not None:
            self.app.request_render()

    def walk(self) -> Iterator["Widget"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def query_one(self, widget_id: str) -> "Widget | None":
        for widget in self.walk():
            if widget.id == widget_id:
                return widget
        return None

    def query_all(self, widget_type: type | tuple[type, ...]) -> list["Widget"]:
        return [widget for widget in self.walk() if isinstance(widget, widget_type)]

    def find(self, predicate: Callable[["Widget"], bool]) -> list["Widget"]:
        return [widget for widget in self.walk() if predicate(widget)]

    # ------------------------------------------------------------------
    # 渲染 / 事件
    # ------------------------------------------------------------------

    def render(self, width: int, height: int) -> list[Line]:
        return [blank_line(width, self.base_style) for _ in range(height)]

    def handle_key(self, key: Key) -> bool:
        return False

    def handle_mouse(self, event: MouseEvent) -> bool:
        return False

    def on_focus(self) -> None:
        pass

    def on_blur(self) -> None:
        pass

    def content_size(self) -> tuple[int, int]:
        return (0, 0)

    def natural_size(self, width: int) -> tuple[int, int]:
        """按实际宽度估算自然尺寸（默认回退 content_size，可被子类覆盖）。"""
        return self.content_size()

    def cursor_position(self) -> tuple[int, int] | None:
        return None

    def post_message(self, message: Message, namespace: str = "") -> None:
        if self.app is not None:
            self.app.dispatch_message(message, namespace)

    def focus(self) -> None:
        if self.app is not None:
            self.app.focus(self)
        else:
            self.focused = True

    def blur(self) -> None:
        self.focused = False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"

    def invalidate(self) -> None:
        """清除组件内部缓存（框架级缓存每帧重建，无需额外动作）。"""

    def layout_node(self) -> StackLayoutNode | ScrollLayoutNode | None:
        return None


class Container(Widget):
    """子组件容器：vertical / horizontal 布局（Flex stack 节点）。"""

    def __init__(
        self,
        *,
        direction: str = "vertical",
        gap: int = 0,
        align: str = "stretch",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.direction = direction
        self.gap = max(0, int(gap))
        self.align = align if align in ("stretch", "start", "center", "end") else "stretch"
        self._layout_entries: list[StackLayoutEntry] = []

    def mount(
        self,
        child: Widget,
        *,
        basis: int | str | None = None,
        grow: int | None = None,
        shrink: int | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        visible: Callable[[Any], bool] | None = None,
    ) -> Widget:
        """挂载子组件并登记 flex 选项；未显式提供时按旧 width/height spec 推导。"""
        if child in self.children:
            self.children.remove(child)
            self._layout_entries = [e for e in self._layout_entries if e.component is not child]
        child.parent = self
        self.children.append(child)
        for descendant in child.walk():
            descendant.app = self.app
        spec = child.height_spec if self.direction == "vertical" else child.width_spec
        options: dict[str, Any] = {}
        if basis is not None:
            options["basis"] = basis
        if grow is not None:
            options["grow"] = grow
        if shrink is not None:
            options["shrink"] = shrink
        if min_size is not None:
            options["min_size"] = min_size
        if max_size is not None:
            options["max_size"] = max_size
        if not options:
            if isinstance(spec, int):
                options = {"basis": spec, "shrink": 0}
            elif spec == "auto":
                options = {"basis": "auto", "shrink": 0}
            elif isinstance(spec, str) and spec.endswith("fr"):
                try:
                    fraction = max(1, int(spec[:-2] or "1"))
                except ValueError:
                    fraction = 1
                options = {"basis": 0, "grow": fraction, "shrink": 1, "min_size": 0}
        self._layout_entries.append(StackLayoutEntry(component=child, **options))
        self.refresh()
        return child

    def remove(self, child: Widget | None = None) -> None:
        if child is None:
            if self.parent is not None:
                self.parent.remove(self)
            return
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self._layout_entries = [e for e in self._layout_entries if e.component is not child]
            for descendant in child.walk():
                descendant.app = None
            self.refresh()

    def clear(self) -> None:
        for child in list(self.children):
            self.remove(child)

    def set_child_basis(self, child: Widget, basis: int | str) -> None:
        """动态调整子组件的 flex basis（编辑器补全展开时增高）。"""
        for entry in self._layout_entries:
            if entry.component is child:
                entry.basis = basis
                self.refresh()
                return

    def layout_node(self) -> StackLayoutNode | None:
        return StackLayoutNode(
            type="vstack" if self.direction == "vertical" else "hstack",
            entries=tuple(self._layout_entries),
            gap=self.gap,
            align=self.align,
        )

    def natural_size(self, width: int) -> tuple[int, int]:
        """按实际宽度递归估算自然尺寸（消息换行高度不再被 1000 列误导）。"""
        if self.direction == "vertical":
            height = 0
            child_width = 0
            for child in self.children:
                if not child.visible:
                    continue
                child_w, child_h = _natural_size_of(child, width)
                child_width = max(child_width, child_w)
                height += child_h
            return (max(1, int(width)), height)
        total_width = 0
        height = 0
        for child in self.children:
            if not child.visible:
                continue
            child_w, child_h = _natural_size_of(child, width)
            total_width += child_w
            height = max(height, child_h)
        return (max(1, total_width), height)

    def _allocate(self, total: int, dimension: str) -> list[tuple[Widget, int, int]]:
        """分配子组件尺寸：返回 [(child, start, size)]。"""
        specs = [
            (
                child,
                child.width_spec if dimension == "width" else child.height_spec,
            )
            for child in self.children
            if child.visible
        ]
        fixed: list[tuple[Widget, int]] = []
        auto: list[Widget] = []
        fractions: list[tuple[Widget, int]] = []
        used = 0
        for child, spec in specs:
            if isinstance(spec, int):
                size = max(0, min(spec, max(0, total - used)))
                fixed.append((child, size))
                used += size
            elif spec == "auto":
                auto.append(child)
            else:
                fraction = 1
                if isinstance(spec, str) and spec.endswith("fr"):
                    try:
                        fraction = max(1, int(spec[:-2] or "1"))
                    except ValueError:
                        fraction = 1
                fractions.append((child, fraction))
        remaining = max(0, total - used)
        auto_used = 0
        auto_sizes: list[tuple[Widget, int]] = []
        for child in auto:
            natural = self._natural_size(child, dimension)
            size = min(natural, max(0, remaining - auto_used))
            auto_sizes.append((child, size))
            auto_used += size
        remaining = max(0, remaining - auto_used)
        total_fraction = sum(fraction for _, fraction in fractions) or 1
        fraction_sizes: list[tuple[Widget, int]] = []
        allocated = 0
        for index, (child, fraction) in enumerate(fractions):
            size = remaining * fraction // total_fraction
            if index == len(fractions) - 1:
                size = max(0, remaining - allocated)
            fraction_sizes.append((child, size))
            allocated += size
        size_by_child = {id(child): size for child, size in [*fixed, *auto_sizes, *fraction_sizes]}
        # 保持子组件挂载顺序（fixed/auto/fr 只决定大小，不改变顺序）。
        result: list[tuple[Widget, int, int]] = []
        start = 0
        for child, _spec in specs:
            size = size_by_child[id(child)]
            result.append((child, start, size))
            start += size
        return result

    @staticmethod
    def _natural_size(child: Widget, dimension: str) -> int:
        width, height = child.content_size()
        return width if dimension == "width" else height

    def render(self, width: int, height: int) -> list[Line]:
        return render_layout_frame(self, width, height).lines

    def content_size(self) -> tuple[int, int]:
        if self.direction == "vertical":
            width = 0
            height = 0
            for child in self.children:
                if not child.visible:
                    continue
                child_width, child_height = child.content_size()
                width = max(width, child_width)
                height += child_height
            return (width, height)
        width = 0
        height = 0
        for child in self.children:
            if not child.visible:
                continue
            child_width, child_height = child.content_size()
            width += child_width
            height = max(height, child_height)
        return (width, height)


class Vertical(Container):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(direction="vertical", **kwargs)


class Horizontal(Container):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(direction="horizontal", **kwargs)


class VStack(Vertical):
    pass


class HStack(Horizontal):
    pass


class Box(Container):
    pass


class Spacer(Widget):
    def __init__(
        self,
        *,
        height: int | str = "1fr",
        width: int | str = "1fr",
        **kwargs: Any,
    ) -> None:
        super().__init__(height=height, width=width, **kwargs)

    def render(self, width: int, height: int) -> list[Line]:
        return [blank_line(width, self.base_style) for _ in range(height)]

    def content_size(self) -> tuple[int, int]:
        height = self.height_spec if isinstance(self.height_spec, int) else 0
        return (0, max(0, height))


class Static(Widget):
    """文本静态组件（Rich markup 或纯文本）。"""

    def __init__(self, content: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._content = content

    @property
    def content(self) -> str:
        return self._content

    def update(self, content: str = "") -> None:
        if content != self._content:
            self._content = content
            self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        lines = render_markup(self._content, width, self.base_style)
        if len(lines) > height:
            lines = lines[:height]
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        return lines

    def content_size(self) -> tuple[int, int]:
        if not self._content.strip():
            return (0, 0)
        lines = render_markup(self._content, 1000)
        return (max((len(line) for line in lines), default=0), len(lines))


class Label(Static):
    pass


class Text(Static):
    pass


class Input(Widget):
    """单行输入框（选择器搜索 / 对话框）。"""

    class Changed(Message):
        def __init__(self, input: "Input", value: str) -> None:
            super().__init__()
            self.input = input
            self.value = value

    class Submitted(Message):
        def __init__(self, input: "Input", value: str) -> None:
            super().__init__()
            self.input = input
            self.value = value

    class Cancelled(Message):
        def __init__(self, input: "Input") -> None:
            super().__init__()
            self.input = input

    def __init__(
        self,
        *,
        value: str = "",
        placeholder: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(focusable=True, **kwargs)
        self.value = value
        self.placeholder = placeholder
        self.cursor = len(value)

    def handle_key(self, key: Key) -> bool:
        if key.name == "enter":
            self.post_message(Input.Submitted(self, self.value), "input")
            return True
        if key.name == "escape":
            self.post_message(Input.Cancelled(self), "input")
            return True
        if key.name == "backspace":
            if self.cursor > 0:
                self.value = self.value[: self.cursor - 1] + self.value[self.cursor :]
                self.cursor -= 1
                self._changed()
            return True
        if key.name == "delete":
            if self.cursor < len(self.value):
                self.value = self.value[: self.cursor] + self.value[self.cursor + 1 :]
                self._changed()
            return True
        if key.name == "left" or key.name == "ctrl+b":
            self.cursor = max(0, self.cursor - 1)
            return True
        if key.name == "right" or key.name == "ctrl+f":
            self.cursor = min(len(self.value), self.cursor + 1)
            return True
        if key.name == "home" or key.name == "ctrl+a":
            self.cursor = 0
            return True
        if key.name == "end" or key.name == "ctrl+e":
            self.cursor = len(self.value)
            return True
        if key.name == "ctrl+u":
            self.value = self.value[self.cursor :]
            self.cursor = 0
            self._changed()
            return True
        if key.char is not None and key.char.isprintable():
            self.value = self.value[: self.cursor] + key.char + self.value[self.cursor :]
            self.cursor += len(key.char)
            self._changed()
            return True
        return False

    def _changed(self) -> None:
        self.refresh()
        self.post_message(Input.Changed(self, self.value), "input")

    def render(self, width: int, height: int) -> list[Line]:
        display = self.value if self.value else self.placeholder
        style = self.base_style
        if not self.value and self.placeholder:
            style = style + Style(dim=True) if style else Style(dim=True)
        line = line_from_text(display, width, style)
        if self.focused and self.value and 0 <= self.cursor < width:
            cell = line.cells[self.cursor]
            cell.style = (cell.style or Style()) + Style(reverse=True)
        return [line]


class Editor(Widget):
    """多行编辑器：vim 模式 / undo / kill ring / word navigation / 选区。"""

    class Submitted(Message):
        def __init__(self, editor: "Editor", text: str) -> None:
            super().__init__()
            self.editor = editor
            self.text = text

    class AutocompleteRequested(Message):
        def __init__(self, editor: "Editor") -> None:
            super().__init__()
            self.editor = editor

    class ExitRequested(Message):
        pass

    class CopyRequested(Message):
        pass

    class CycleThinkingRequested(Message):
        pass

    class CompletionNavigateRequested(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class CompletionSubmitRequested(Message):
        pass

    class CompletionHideRequested(Message):
        pass

    class ModeChanged(Message):
        def __init__(self, editor: "Editor", mode: str) -> None:
            super().__init__()
            self.editor = editor
            self.mode = mode

    def __init__(
        self,
        *,
        text: str = "",
        border: bool = False,
        padding_x: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(focusable=True, **kwargs)
        self.border = border
        self.padding_x = max(0, int(padding_x))
        self.border_style: Style | None = None
        self.lines: list[str] = [""] if not text else text.split("\n")
        self.cursor_row = 0
        self.cursor_col = 0
        self.scroll_row = 0
        self.scroll_col = 0
        self.completion_active = False
        self.vim_mode = "insert"
        self.vim_enabled = False
        self._pending = ""
        self._undo_stack: list[tuple[list[str], int, int]] = []
        self._redo_stack: list[tuple[list[str], int, int]] = []
        self.kill_ring: list[str] = []
        self._jump_mode: str | None = None
        self.selection_anchor: tuple[int, int] | None = None
        self.history: list[str] = []
        self.history_index = -1  # -1=未浏览，0=最近一条
        self.history_draft: tuple[list[str], int, int] | None = None
        self.completion_items: list[tuple[str, str]] = []
        self.completion_index = 0
        self.completion_max_visible = 5

    # ------------------------------------------------------------------
    # 文本属性（兼容 PiEditor API）
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @text.setter
    def text(self, value: str) -> None:
        self._snapshot()
        self.lines = value.split("\n") if value else [""]
        self.cursor_row = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self.selection_anchor = None
        self._exit_history_browsing()
        self._clamp_cursor()
        self.refresh()

    def clear(self) -> None:
        self._snapshot()
        self.lines = [""]
        self.cursor_row = 0
        self.cursor_col = 0
        self.selection_anchor = None
        self._pending = ""
        self._exit_history_browsing()
        self.refresh()

    def insert(self, value: str) -> None:
        """在光标处插入文本（含换行）。"""
        self._snapshot()
        self.selection_anchor = None
        self._exit_history_browsing()
        if "\n" in value:
            head, _, tail = value.partition("\n")
            current = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = current[: self.cursor_col] + head
            new_lines = tail.split("\n")
            self.lines[self.cursor_row + 1 : self.cursor_row + 1] = new_lines
            self.cursor_row += len(new_lines)
            self.cursor_col = len(new_lines[-1])
        else:
            current = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = (
                current[: self.cursor_col] + value + current[self.cursor_col :]
            )
            self.cursor_col += len(value)
        self._clamp_cursor()
        self.refresh()

    @property
    def selection(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        bounds = self._selection_bounds()
        start, end = bounds
        if start is None or end is None:
            return None
        return (start, end)

    @property
    def selected_text(self) -> str:
        start, end = self._selection_bounds()
        if start is None or end is None:
            return ""
        if start[0] == end[0]:
            return self.lines[start[0]][start[1] : end[1]]
        parts = [self.lines[start[0]][start[1] :]]
        for row in range(start[0] + 1, end[0]):
            parts.append(self.lines[row])
        parts.append(self.lines[end[0]][: end[1]])
        return "\n".join(parts)

    def _selection_bounds(self) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if self.selection_anchor is None:
            return None, None
        anchor = self.selection_anchor
        cursor = (self.cursor_row, self.cursor_col)
        if anchor <= cursor:
            return anchor, cursor
        return cursor, anchor

    def undo(self) -> None:
        if not self._undo_stack:
            return
        lines, row, col = self._undo_stack.pop()
        self._redo_stack.append((self.lines, self.cursor_row, self.cursor_col))
        self.lines = lines
        self.cursor_row = row
        self.cursor_col = col
        self.selection_anchor = None
        self._clamp_cursor()
        self.refresh()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        lines, row, col = self._redo_stack.pop()
        self._undo_stack.append((self.lines, self.cursor_row, self.cursor_col))
        self.lines = lines
        self.cursor_row = row
        self.cursor_col = col
        self.selection_anchor = None
        self._clamp_cursor()
        self.refresh()

    def _snapshot(self) -> None:
        self._undo_stack.append(([line for line in self.lines], self.cursor_row, self.cursor_col))
        if len(self._undo_stack) > 500:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _clamp_cursor(self) -> None:
        if not self.lines:
            self.lines = [""]
        self.cursor_row = max(0, min(self.cursor_row, len(self.lines) - 1))
        self.cursor_col = max(0, min(self.cursor_col, len(self.lines[self.cursor_row])))

    # ------------------------------------------------------------------
    # 光标 / 编辑
    # ------------------------------------------------------------------

    def move_cursor(self, location: tuple[int, int]) -> None:
        self.cursor_row, self.cursor_col = location
        self.selection_anchor = None
        self._clamp_cursor()
        self.refresh()

    def move_cursor_relative(self, *, rows: int = 0, columns: int = 0) -> None:
        row = self.cursor_row + rows
        row = max(0, min(row, len(self.lines) - 1))
        col = self.cursor_col + columns
        if columns:
            col = max(0, min(col, len(self.lines[row])))
        self.move_cursor((row, col))

    def _select_move(self, rows: int = 0, columns: int = 0) -> None:
        """Shift 移动：保持并扩展选区。"""
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        row = self.cursor_row + rows
        row = max(0, min(row, len(self.lines) - 1))
        col = self.cursor_col + columns
        if columns:
            col = max(0, min(col, len(self.lines[row])))
        self.cursor_row = row
        self.cursor_col = col
        self.refresh()

    def _select_home(self) -> None:
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        self.cursor_col = 0
        self.refresh()

    def _select_end(self) -> None:
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        self.cursor_col = len(self.lines[self.cursor_row])
        self.refresh()

    def _select_document_start(self) -> None:
        """ctrl+shift+home：选区扩展到文档首（对齐 TS）。"""
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        self.cursor_row = 0
        self.cursor_col = 0
        self.refresh()

    def _select_document_end(self) -> None:
        """ctrl+shift+end：选区扩展到文档尾（对齐 TS）。"""
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        self.cursor_row = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self.refresh()

    def _select_word(self, direction: int) -> None:
        """ctrl+shift+方向：按词扩展选区（不把分隔符并入词尾）。"""
        if self.selection_anchor is None:
            self.selection_anchor = (self.cursor_row, self.cursor_col)
        text = self.lines[self.cursor_row]
        if direction < 0:
            self.cursor_col = self._word_start(self.cursor_row, self.cursor_col)
        else:
            index = self.cursor_col
            length = len(text)
            while index < length and not (text[index].isalnum() or text[index] == "_"):
                index += 1
            while index < length and (text[index].isalnum() or text[index] == "_"):
                index += 1
            self.cursor_col = index
        self.refresh()

    def delete(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """删除 [start, end) 区间（兼容旧 API）。"""
        self._snapshot()
        self.selection_anchor = None
        if start[0] == end[0]:
            row = start[0]
            self.lines[row] = self.lines[row][: start[1]] + self.lines[row][end[1] :]
            self.cursor_row, self.cursor_col = start
        elif start[0] < end[0]:
            first = self.lines[start[0]][: start[1]]
            last = self.lines[end[0]][end[1] :]
            self.lines[start[0] : end[0] + 1] = [first + last]
            self.cursor_row, self.cursor_col = start
        self._clamp_cursor()
        self.refresh()

    def _insert_newline(self) -> None:
        self.insert("\n")

    def _delete_char_before(self) -> None:
        if self.cursor_col > 0:
            self.delete(
                (self.cursor_row, self.cursor_col - 1),
                (self.cursor_row, self.cursor_col),
            )
        elif self.cursor_row > 0:
            previous = self.lines[self.cursor_row - 1]
            self._snapshot()
            self.lines[self.cursor_row - 1] = previous + self.lines[self.cursor_row]
            self.lines.pop(self.cursor_row)
            self.cursor_row -= 1
            self.cursor_col = len(previous)
            self.selection_anchor = None
            self.refresh()

    def _delete_char_after(self) -> None:
        row = self.cursor_row
        if self.cursor_col < len(self.lines[row]):
            self.delete((row, self.cursor_col), (row, self.cursor_col + 1))
        elif row < len(self.lines) - 1:
            self._snapshot()
            self.lines[row] = self.lines[row] + self.lines[row + 1]
            self.lines.pop(row + 1)
            self.selection_anchor = None
            self.refresh()

    def _delete_line(self) -> None:
        row = self.cursor_row
        self._snapshot()
        self.selection_anchor = None
        if len(self.lines) == 1:
            self.lines[0] = ""
            self.cursor_col = 0
        else:
            self.lines.pop(row)
            self.cursor_row = min(row, len(self.lines) - 1)
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
        self.refresh()

    def _delete_to_line_start(self) -> None:
        col = self.cursor_col
        if col > 0:
            self.delete((self.cursor_row, 0), (self.cursor_row, col))
            self.move_cursor((self.cursor_row, 0))

    def _delete_word_before(self) -> None:
        col = self._word_start(self.cursor_row, self.cursor_col)
        if col < self.cursor_col:
            self.delete((self.cursor_row, col), (self.cursor_row, self.cursor_col))
            self.move_cursor((self.cursor_row, col))

    def _delete_word_after(self) -> None:
        """alt+d / alt+delete：删除光标后的一个词（对齐 TS deleteWordForward）。"""
        index = self._word_end(self.cursor_row, self.cursor_col)
        if index > self.cursor_col:
            self.delete(
                (self.cursor_row, self.cursor_col),
                (self.cursor_row, index),
            )

    def _yank_pop(self) -> None:
        """alt+y：kill ring 循环回退（对齐 TS yankPop）。"""
        if not self.kill_ring:
            return
        self.kill_ring.insert(0, self.kill_ring.pop())
        self._yank()

    def _jump_to_char(self, char: str, direction: str) -> None:
        """ctrl+] / ctrl+alt+]：跳到下一个/上一个指定字符。"""
        if not char:
            return
        if direction == "forward":
            for row in range(self.cursor_row, len(self.lines)):
                start = self.cursor_col + 1 if row == self.cursor_row else 0
                index = self.lines[row].find(char, start)
                if index != -1:
                    self.move_cursor((row, index))
                    return
        else:
            for row in range(self.cursor_row, -1, -1):
                end = self.cursor_col if row == self.cursor_row else len(self.lines[row])
                index = self.lines[row].rfind(char, 0, end)
                if index != -1:
                    self.move_cursor((row, index))
                    return

    def _kill_to_line_end(self) -> None:
        """ctrl+k：把光标到行尾的文本压入 kill ring 并删除。"""
        row = self.cursor_row
        line = self.lines[row]
        if self.cursor_col < len(line):
            killed = line[self.cursor_col :]
            self.kill_ring.append(killed)
            self.delete((row, self.cursor_col), (row, len(line)))
        elif row < len(self.lines) - 1:
            self.kill_ring.append("")
            self._snapshot()
            self.lines[row] = line + self.lines[row + 1]
            self.lines.pop(row + 1)
            self.selection_anchor = None
            self.refresh()

    def _yank(self) -> None:
        """ctrl+y：粘贴最近一次 kill。"""
        if self.kill_ring:
            self.insert(self.kill_ring[-1])

    def _word_start(self, row: int, col: int) -> int:
        text = self.lines[row]
        index = col
        while index > 0 and not text[index - 1].isalnum() and text[index - 1] != "_":
            index -= 1
        while index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_"):
            index -= 1
        return index

    def _word_end(self, row: int, col: int) -> int:
        text = self.lines[row]
        index = col
        length = len(text)
        while index < length and (text[index].isalnum() or text[index] == "_"):
            index += 1
        while index < length and not text[index].isalnum() and text[index] != "_":
            index += 1
        return index

    def _move_word_forward(self) -> None:
        col = self._word_end(self.cursor_row, self.cursor_col)
        self.move_cursor((self.cursor_row, col))

    def _move_word_backward(self) -> None:
        col = self._word_start(self.cursor_row, self.cursor_col)
        self.move_cursor((self.cursor_row, col))

    def _open_line_below(self) -> None:
        row = self.cursor_row
        self._snapshot()
        self.lines.insert(row + 1, "")
        self.cursor_row = row + 1
        self.cursor_col = 0
        self.selection_anchor = None
        self.toggle_mode()
        self.refresh()

    def toggle_mode(self) -> None:
        self.vim_mode = "normal" if self.vim_mode == "insert" else "insert"
        self._pending = ""
        self.selection_anchor = None
        self.post_message(Editor.ModeChanged(self, self.vim_mode), "pi_editor")

    # ------------------------------------------------------------------
    # 输入历史（对齐 TS Editor：上下键召回已提交的 prompt）
    # ------------------------------------------------------------------

    def add_to_history(self, text: str) -> None:
        """提交成功后记录输入（去重、上限 100）。"""
        trimmed = text.strip()
        if not trimmed:
            return
        if self.history and self.history[0] == trimmed:
            return
        self.history.insert(0, trimmed)
        if len(self.history) > 100:
            self.history.pop()

    def navigate_history(self, direction: int) -> None:
        """direction=-1 更旧（上），1 更新（下）。"""
        if not self.history:
            return
        new_index = self.history_index - direction
        if new_index < -1 or new_index >= len(self.history):
            return
        if self.history_index == -1 and new_index >= 0:
            self._snapshot()
            self.history_draft = (
                [line for line in self.lines],
                self.cursor_row,
                self.cursor_col,
            )
        self.history_index = new_index
        if self.history_index == -1:
            draft = self.history_draft
            self.history_draft = None
            if draft is not None:
                self.lines, self.cursor_row, self.cursor_col = draft
            else:
                self.lines = [""]
                self.cursor_row = 0
                self.cursor_col = 0
            self.selection_anchor = None
            self._clamp_cursor()
        else:
            text = self.history[self.history_index] or ""
            self.lines = text.split("\n") or [""]
            self.cursor_row = 0 if direction == -1 else len(self.lines) - 1
            self.cursor_col = 0 if direction == -1 else len(self.lines[self.cursor_row])
            self.selection_anchor = None
            self._clamp_cursor()
        self.refresh()

    def _exit_history_browsing(self) -> None:
        self.history_index = -1
        self.history_draft = None

    # ------------------------------------------------------------------
    # 按键
    # ------------------------------------------------------------------

    def handle_key(self, key: Key) -> bool:
        if self.vim_mode == "normal":
            return self._handle_normal_key(key)
        return self._handle_insert_key(key)

    def _handle_insert_key(self, key: Key) -> bool:
        name = key.name
        if self.completion_active:
            if name == "up":
                self.post_message(Editor.CompletionNavigateRequested(-1), "pi_editor")
                return True
            if name == "down":
                self.post_message(Editor.CompletionNavigateRequested(1), "pi_editor")
                return True
            if name == "escape":
                self.post_message(Editor.CompletionHideRequested(), "pi_editor")
                return True
        if name == "enter":
            if self.completion_active:
                self.post_message(Editor.CompletionSubmitRequested(), "pi_editor")
            else:
                text = self.text.strip()
                if text:
                    self.post_message(Editor.Submitted(self, text), "pi_editor")
                    self.clear()
            return True
        if name == "escape":
            if self.vim_enabled:
                self.toggle_mode()
                return True
            return False
        if name == "shift+enter":
            self._insert_newline()
            return True
        if name == "tab":
            self.post_message(Editor.AutocompleteRequested(self), "pi_editor")
            return True
        if name == "backspace":
            self._delete_char_before()
            return True
        if name == "delete":
            self._delete_char_after()
            return True
        if name == "left" or name == "ctrl+b":
            self.move_cursor_relative(columns=-1)
            return True
        if name == "right" or name == "ctrl+f":
            self.move_cursor_relative(columns=1)
            return True
        if name == "up":
            if self.cursor_row == 0 and (self.history_index != -1 or self.cursor_col == 0):
                self.navigate_history(-1)
            elif self.cursor_row == 0:
                self.cursor_col = 0
                self.selection_anchor = None
                self.refresh()
            else:
                self.move_cursor_relative(rows=-1)
            return True
        if name == "down":
            if self.history_index != -1 and self.cursor_row == len(self.lines) - 1:
                self.navigate_history(1)
            elif self.cursor_row == len(self.lines) - 1:
                self.cursor_col = len(self.lines[self.cursor_row])
                self.selection_anchor = None
                self.refresh()
            else:
                self.move_cursor_relative(rows=1)
            return True
        if name == "shift+left":
            self._select_move(columns=-1)
            return True
        if name == "shift+right":
            self._select_move(columns=1)
            return True
        if name == "shift+up":
            self._select_move(rows=-1)
            return True
        if name == "shift+down":
            self._select_move(rows=1)
            return True
        if name == "shift+home":
            self._select_home()
            return True
        if name == "shift+end":
            self._select_end()
            return True
        if name == "ctrl+shift+home":
            self._select_document_start()
            return True
        if name == "ctrl+shift+end":
            self._select_document_end()
            return True
        if name == "home":
            self.cursor_col = 0
            self.selection_anchor = None
            self.refresh()
            return True
        if name == "end":
            self.cursor_col = len(self.lines[self.cursor_row])
            self.selection_anchor = None
            self.refresh()
            return True
        if name == "pageup":
            self.move_cursor_relative(rows=-6)
            return True
        if name == "pagedown":
            self.move_cursor_relative(rows=6)
            return True
        if name in ("ctrl+left", "alt+left", "alt+b"):
            self._move_word_backward()
            return True
        if name in ("ctrl+right", "alt+right", "alt+f"):
            self._move_word_forward()
            return True
        if name in ("ctrl+w", "alt+backspace"):
            self._delete_word_before()
            return True
        if name in ("alt+d", "alt+delete"):
            self._delete_word_after()
            return True
        if name == "ctrl+u":
            self._delete_to_line_start()
            return True
        if name == "ctrl+shift+left":
            self._select_word(-1)
            return True
        if name == "ctrl+shift+right":
            self._select_word(1)
            return True
        if name == "ctrl+k":
            self._kill_to_line_end()
            return True
        if name == "ctrl+y":
            self._yank()
            return True
        if name == "alt+y":
            self._yank_pop()
            return True
        if name == "ctrl+-":
            self.undo()
            return True
        if name == "ctrl+]":
            self._jump_mode = "forward"
            return True
        if name == "ctrl+alt+]":
            self._jump_mode = "backward"
            return True
        if name == "ctrl+d":
            if not self.text:
                self.post_message(Editor.ExitRequested(), "pi_editor")
            else:
                self._delete_char_after()
            return True
        if name == "ctrl+x":
            self.post_message(Editor.CopyRequested(), "pi_editor")
            return True
        if name == "shift+tab":
            self.post_message(Editor.CycleThinkingRequested(), "pi_editor")
            return True
        if name == "ctrl+c":
            selected = self.selected_text
            if selected and self.app is not None and hasattr(self.app, "copy_to_clipboard"):
                self.app.copy_to_clipboard(selected)
            else:
                self.clear()
            return True
        if key.char is not None and key.char.isprintable():
            if self._jump_mode is not None:
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(key.char, direction)
                return True
            self.insert(key.char)
            if self.text.startswith("/") and " " not in self.text:
                self.post_message(Editor.AutocompleteRequested(self), "pi_editor")
            return True
        self._jump_mode = None
        return False

    def _handle_normal_key(self, key: Key) -> bool:
        name = key.name
        if name == "escape":
            self.toggle_mode()
            return True
        if name == "enter":
            text = self.text.strip()
            if text:
                self.post_message(Editor.Submitted(self, text), "pi_editor")
                self.clear()
            return True
        if name == "i":
            self.toggle_mode()
            return True
        if name == "a":
            self.move_cursor_relative(columns=1)
            self.toggle_mode()
            return True
        if name == "o":
            self._open_line_below()
            return True
        if name == "h":
            self.move_cursor_relative(columns=-1)
        elif name == "l":
            self.move_cursor_relative(columns=1)
        elif name == "j":
            self.move_cursor_relative(rows=1)
        elif name == "k":
            self.move_cursor_relative(rows=-1)
        elif name == "0":
            self.cursor_col = 0
            self.selection_anchor = None
            self.refresh()
        elif name == "$":
            self.cursor_col = len(self.lines[self.cursor_row])
            self.selection_anchor = None
            self.refresh()
        elif name == "x":
            self._delete_char_after()
        elif name == "u":
            self.undo()
        elif name == "d":
            if self._pending == "d":
                self._pending = ""
                self._delete_line()
            else:
                self._pending = "d"
            return True
        else:
            self._pending = ""
            return False
        self._pending = ""
        return True

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(self, width: int, height: int) -> list[Line]:
        border_offset = 1 if self.border else 0
        completion_count = self._completion_line_count()
        content_height = max(1, height - 2 * border_offset - completion_count)
        if self.cursor_row < self.scroll_row:
            self.scroll_row = self.cursor_row
        elif self.cursor_row >= self.scroll_row + content_height:
            self.scroll_row = self.cursor_row - content_height + 1
        if self.cursor_col < self.scroll_col:
            self.scroll_col = self.cursor_col
        elif self.cursor_col >= self.scroll_col + width:
            self.scroll_col = self.cursor_col - width + 1
        content: list[Line] = []
        start, end = self._selection_bounds()
        for row in range(self.scroll_row, min(len(self.lines), self.scroll_row + content_height)):
            raw = self.lines[row][
                self.scroll_col : self.scroll_col + max(0, width - self.padding_x)
            ]
            text = " " * self.padding_x + raw
            line = line_from_text(text, width, self.base_style)
            if (
                self.focused
                and row == self.cursor_row
                and self.padding_x + self.cursor_col - self.scroll_col < width
            ):
                index = self.padding_x + self.cursor_col - self.scroll_col
                cell = line.cells[index]
                cell.style = (cell.style or Style()) + Style(reverse=True)
            if start is not None and end is not None and start[0] <= row <= end[0]:
                from_col = start[1] if row == start[0] else 0
                to_col = end[1] if row == end[0] else len(self.lines[row])
                from_col = max(0, self.padding_x + from_col - self.scroll_col)
                to_col = min(width, max(0, self.padding_x + to_col - self.scroll_col))
                for index in range(from_col, to_col):
                    cell = line.cells[index]
                    cell.style = (cell.style or Style()) + Style(reverse=True)
            content.append(line)
        while len(content) < content_height:
            content.append(blank_line(width, self.base_style))
        if self.border:
            border_style = self.border_style or (self.base_style or Style()) + Style(dim=True)
            border = line_from_text("─" * width, width, border_style)
            result = [border, *content, border]
        else:
            result = content
        if completion_count:
            result.extend(self._completion_lines(width))
        return result[:height]

    def cursor_position(self) -> tuple[int, int] | None:
        border_offset = 1 if self.border else 0
        return (
            border_offset + self.cursor_row - self.scroll_row,
            self.padding_x + self.cursor_col - self.scroll_col,
        )

    def content_size(self) -> tuple[int, int]:
        return (
            max((len(line) for line in self.lines), default=0),
            len(self.lines) + self._completion_line_count(),
        )

    def _completion_line_count(self) -> int:
        """补全列表占用的行数（可见项 + 滚动指示，未激活时 0）。"""
        if not self.completion_active or not self.completion_items:
            return 0
        count = min(len(self.completion_items), self.completion_max_visible)
        if len(self.completion_items) > self.completion_max_visible:
            count += 1
        return count

    def set_completion(
        self,
        items: list[tuple[str, str]],
        index: int = 0,
        max_visible: int = 5,
    ) -> None:
        """设置编辑器内嵌补全列表（对齐 TS editor 下方的 SelectList）。"""
        self.completion_items = list(items)
        self.completion_max_visible = max(1, int(max_visible))
        self.completion_index = max(0, min(int(index), max(0, len(self.completion_items) - 1)))
        self.completion_active = bool(self.completion_items)
        self.refresh()

    def clear_completion(self) -> None:
        self.completion_items = []
        self.completion_index = 0
        self.completion_active = False
        self.refresh()

    def _completion_lines(self, width: int) -> list[Line]:
        """渲染补全列表（→ 选中标记 + 值列 + dim 描述 + 滚动指示）。"""
        items = self.completion_items
        max_visible = self.completion_max_visible
        start = max(
            0,
            min(
                self.completion_index - max_visible // 2,
                len(items) - max_visible,
            ),
        )
        end = min(start + max_visible, len(items))
        column_width = max((len(value) for value, _label in items), default=0) + 2
        base = self.base_style
        lines: list[Line] = []
        for index in range(start, end):
            value, label = items[index]
            selected = index == self.completion_index
            prefix = "→ " if selected else "  "
            spacing = " " * max(1, column_width - len(value))
            line = line_from_text(f"{prefix}{value}{spacing}{label}", width, base)
            if selected:
                for cell in line.cells:
                    cell.style = (cell.style or Style()) + Style(reverse=True)
            elif label:
                desc_start = len(prefix) + len(value) + len(spacing)
                for cell in line.cells[desc_start : desc_start + len(label)]:
                    cell.style = (cell.style or Style()) + Style(dim=True)
            lines.append(line)
        if len(items) > max_visible:
            info = f"  ({self.completion_index + 1}/{len(items)})"
            lines.append(line_from_text(info, width, (base or Style()) + Style(dim=True)))
        return lines

    def handle_mouse(self, event: MouseEvent) -> bool:
        if event.type == "wheel" and self.rect[2] > 0:
            self.move_cursor_relative(rows=-1 if event.button == "up" else 1)
            return True
        return False


class PiEditor(Editor):
    """兼容别名：引擎 Editor 即 PiEditor。"""


class PiEditorVim(Editor):
    """vim 模式编辑器（Esc 切换 normal/insert）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vim_mode = "insert"
        self.vim_enabled = True


class ScrollView(Widget):
    """垂直滚动视口（Flex scroll 节点）：滚动状态 + 滚动条（支持拖拽）。"""

    def __init__(
        self,
        child: Widget | None = None,
        *,
        scroll_offset: int = 0,
        follow: str = "none",
        primary: bool = False,
        overscroll: str = "chain",
        scrollbar: str = "hidden",
        scrollbar_style: Style | str | Callable[[Style | None], Style] | None = None,
        scrollbar_hide_delay_ms: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if follow not in ("none", "end"):
            raise ValueError(f"Unsupported ScrollView follow: {follow}")
        if overscroll not in ("chain", "contain"):
            raise ValueError(f"Unsupported ScrollView overscroll: {overscroll}")
        if scrollbar not in ("hidden", "auto", "always"):
            raise ValueError(f"Unsupported ScrollView scrollbar: {scrollbar}")
        self._child = child
        self._scroll_top = max(0, int(scroll_offset))
        self._follow = follow
        self.follow_end = follow == "end"
        self._following_end = self.follow_end
        self.primary = bool(primary)
        self.overscroll = overscroll
        self._scrollbar = scrollbar
        self._scrollbar_style = scrollbar_style
        self._scrollbar_hide_delay_ms = max(0, int(scrollbar_hide_delay_ms))
        self._content_height = 0
        self._viewport_height = 0
        self._request_render: Callable[[], None] | None = None
        self._dragging = False
        self.scrollbar_active = False
        if child is not None:
            child.parent = self

    # ------------------------------------------------------------------
    # 状态属性（scroll_offset 兼容旧 API，scroll_top 对齐 TS）
    # ------------------------------------------------------------------

    @property
    def scroll_offset(self) -> int:
        return self._scroll_top

    @scroll_offset.setter
    def scroll_offset(self, value: int) -> None:
        self._scroll_top = max(0, int(value))

    @property
    def scroll_top(self) -> int:
        return self._scroll_top

    @scroll_top.setter
    def scroll_top(self, value: int) -> None:
        self.scroll_to(int(value))

    @property
    def child(self) -> Widget | None:
        return self._child

    @property
    def viewport_height(self) -> int:
        return self._viewport_height

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def scrollbar(self) -> str:
        return self._scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        if self._scrollbar == "always":
            return self._viewport_height > 0
        if self._scrollbar == "auto":
            return self._content_height > self._viewport_height
        return False

    def set_scrollbar(self, scrollbar: str) -> None:
        if scrollbar == self._scrollbar:
            return
        self._scrollbar = scrollbar
        self.refresh()

    def set_scrollbar_active(self, active: bool) -> None:
        if self.scrollbar_active != bool(active):
            self.scrollbar_active = bool(active)
            self.refresh()

    def scrollbar_style(self, base: Style | None) -> Style:
        style = self._scrollbar_style
        if callable(style):
            return style(base)
        if isinstance(style, Style):
            return (base or Style()) + style
        if isinstance(style, str):
            return (base or Style()) + Style.parse(style)
        return (base or Style()) + Style(bgcolor="bright_black")

    def _content_max(self) -> int:
        """最大滚动偏移：布局状态未知时回退到子组件 content_size。"""
        if self._viewport_height > 0:
            return max(0, self._content_height - self._viewport_height)
        if self._child is not None:
            _, content_height = self._child.content_size()
            return max(0, content_height - self.rect[3])
        return 0

    # ------------------------------------------------------------------
    # 树 / 布局
    # ------------------------------------------------------------------

    def mount_child(self, child: Widget) -> Widget:
        self._child = child
        child.parent = self
        for descendant in child.walk():
            descendant.app = self.app
        self.refresh()
        return child

    def walk(self) -> Iterator["Widget"]:
        yield self
        for child in self.children:
            yield from child.walk()
        if self._child is not None:
            yield from self._child.walk()

    def layout_node(self) -> ScrollLayoutNode | None:
        return ScrollLayoutNode(type="scroll", component=self._child, state=self)

    def get_content_width(self, width: int) -> int:
        return max(1, width - 1) if self._scrollbar == "always" and width > 1 else max(1, width)

    def natural_size(self, width: int) -> tuple[int, int]:
        """滚动视口自然高度 = 子内容高度（regular 文档模式用）。"""
        if self._child is None:
            return (max(1, int(width)), 0)
        content_width = self.get_content_width(width)
        _child_width, height = _natural_size_of(self._child, content_width)
        return (max(1, int(width)), max(0, int(height)))

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        self._content_height = max(0, int(content_height))
        self._viewport_height = max(0, int(viewport_height))
        self._request_render = request_render
        max_scroll_top = max(0, self._content_height - self._viewport_height)
        if self._following_end:
            self._scroll_top = max_scroll_top
        else:
            self._scroll_top = max(0, min(self._scroll_top, max_scroll_top))
        if self.follow_end and self._scroll_top == max_scroll_top:
            self._following_end = True

    def scroll_to(self, scroll_top: int) -> None:
        requested = int(scroll_top)
        max_scroll_top = self._content_max()
        next_value = max(0, min(max_scroll_top, requested))
        if next_value == self._scroll_top:
            return
        self._scroll_top = next_value
        self._following_end = self.follow_end and next_value == max_scroll_top
        self.refresh()

    def scroll_by(self, lines: int) -> int:
        requested = int(lines)
        if requested == 0:
            return 0
        max_scroll_top = self._content_max()
        start = max_scroll_top if self._following_end else self._scroll_top
        next_value = max(0, min(max_scroll_top, start + requested))
        moved = next_value - start
        self._scroll_top = next_value
        self._following_end = self.follow_end and next_value == max_scroll_top
        if moved != 0:
            self.refresh()
        return requested - moved

    def scroll_to_start(self) -> None:
        self._scroll_top = 0
        self._following_end = self.follow_end and self._content_height <= self._viewport_height
        self.refresh()

    def scroll_to_end(self) -> None:
        self._scroll_top = max(0, self._content_height - self._viewport_height)
        self._following_end = self.follow_end
        self.refresh()

    def scroll_end(self) -> None:
        self._scroll_top = self._content_max()
        self.refresh()

    # ------------------------------------------------------------------
    # 按键 / 鼠标
    # ------------------------------------------------------------------

    def handle_key(self, key: Key) -> bool:
        if key.name == "up":
            self._scroll_top = max(0, self._scroll_top - 1)
            self.refresh()
            return True
        if key.name == "down":
            self._scroll_top += 1
            self.refresh()
            return True
        if key.name == "pageup":
            self._scroll_top = max(0, self._scroll_top - self.rect[3])
            self.refresh()
            return True
        if key.name == "pagedown":
            self._scroll_top += self.rect[3]
            self.refresh()
            return True
        return False

    def handle_mouse(self, event: MouseEvent) -> bool:
        if event.type == "wheel":
            self._scroll_top += -3 if event.button == "up" else 3
            self._scroll_top = max(0, self._scroll_top)
            self.refresh()
            return True
        if event.type == "press" and event.button == "left" and self.rect[2] > 0:
            col = event.col - self.rect[1]
            if col >= self.rect[2] - 1:
                self._dragging = True
                self._drag_to_row(event.row)
                if self.app is not None:
                    self.app._drag_target = self
                return True
        if event.type == "motion" and self._dragging:
            self._drag_to_row(event.row)
            return True
        if event.type == "release" and self._dragging:
            self._dragging = False
            return True
        return False

    def _drag_to_row(self, screen_row: int) -> None:
        """把滚动条拖到屏幕行 → 按比例设置偏移。"""
        _row, _col, _width, height = self.rect
        if height <= 1:
            return
        max_scroll_top = self._content_max()
        local = screen_row - self.rect[0]
        ratio = local / max(1, height - 1)
        self._scroll_top = round(ratio * max_scroll_top)
        self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        if self._child is None:
            return [blank_line(width, self.base_style) for _ in range(height)]
        _, content_height = self._child.content_size()
        max_offset = max(0, content_height - height)
        self._scroll_top = min(self._scroll_top, max_offset)
        bar_width = 1 if width > 2 else 0
        view_width = max(1, width - bar_width)
        child_lines = self._child.render(view_width, max(height, content_height))
        lines: list[Line] = []
        for row in range(height):
            index = self._scroll_top + row
            if index < len(child_lines):
                line = child_lines[index].copy()
            else:
                line = blank_line(view_width, self.base_style)
            if bar_width:
                line.patch(
                    view_width,
                    self._scrollbar_cell(row, viewport=height, content=content_height),
                )
            lines.append(line)
        return lines

    def _scrollbar_cell(self, row: int, *, viewport: int, content: int) -> Line:
        """当前行是否属于滚动条拇指（hover 时反色高亮）。"""
        if content <= viewport or viewport <= 0:
            return Line([Cell(" ", self.base_style)])
        thumb = max(1, round(viewport * viewport / content))
        max_position = max(0, viewport - thumb)
        position = round(self._scroll_top / max(1, content - viewport) * max_position)
        char = "█" if position <= row < position + thumb else " "
        style = self.base_style
        if self.scrollbar_active and char == "█":
            style = (style or Style()) + Style(reverse=True)
        return Line([Cell(char, style)])


class Markdown(Widget):
    """Markdown 文本组件。"""

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._text = text

    def update(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        lines = render_markdown(self._text, width)
        return (lines + [blank_line(width, self.base_style) for _ in range(height)])[:height]

    def content_size(self) -> tuple[int, int]:
        return (1000, len(render_markdown(self._text, 1000)))


class Loader(Widget):
    """加载指示器（App 每帧 tick 可推进动画）。"""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, text: str = "Loading…", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text = text
        self.frame = 0

    def tick(self) -> None:
        self.frame += 1
        self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        spinner = self.FRAMES[self.frame % len(self.FRAMES)]
        return [line_from_text(f"{spinner} {self.text}", width, self.base_style)]

    def content_size(self) -> tuple[int, int]:
        return (max(1, len(self.text) + 2), 1)


class CancellableLoader(Loader):
    def __init__(self, text: str = "Loading… (Esc to cancel)", **kwargs: Any) -> None:
        super().__init__(text=text, **kwargs)
        self.cancelled = False

    def handle_key(self, key: Key) -> bool:
        if key.name == "escape":
            self.cancelled = True
            return True
        return False


class AltScreenFlash(Widget):
    """全屏闪烁提示（对齐 TS AltScreenFlash）：居中显示一行文本。"""

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text = text

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = []
        middle = max(0, (height - 1) // 2)
        for row in range(height):
            if row == middle:
                lines.append(line_from_text(self.text, width, self.base_style))
            else:
                lines.append(blank_line(width, self.base_style))
        return lines

    def content_size(self) -> tuple[int, int]:
        return (max(1, len(self.text)), 1)


# ---------------------------------------------------------------------------
# 列表组件（SelectList / SettingsList）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectItem:
    value: str
    label: str | None = None
    description: str | None = None

    @property
    def display_label(self) -> str:
        return self.label or self.value


@dataclass(frozen=True)
class SettingItem:
    id: str
    label: str
    current_value: str = ""
    values: list[str] | None = None
    description: str | None = None


def _normalize_select_items(items: Sequence[SelectItem | str]) -> list[SelectItem]:
    return [item if isinstance(item, SelectItem) else SelectItem(value=str(item)) for item in items]


def _filter_score(text: str, query: str) -> int:
    """模糊匹配打分：2=前缀，1=子串，0=子序列，-1=不匹配。"""
    text = text.lower()
    query = query.strip().lower()
    if not query:
        return 0
    if text.startswith(query):
        return 2
    if query in text:
        return 1
    iterator = iter(text)
    if all(any(ch == wanted for ch in iterator) for wanted in query):
        return 0
    return -1


def _filter_items(items: list[SelectItem], query: str) -> list[SelectItem]:
    scored: list[tuple[int, SelectItem]] = []
    for item in items:
        score = max(_filter_score(item.value, query), _filter_score(item.display_label, query))
        if score >= 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].display_label.lower()))
    return [item for _, item in scored]


class SelectList(Widget):
    """可筛选选项列表：输入即过滤，↑↓ 导航，Enter 选择，Esc 取消。"""

    class Selected(Message):
        def __init__(self, item: SelectItem) -> None:
            super().__init__()
            self.item = item

    class Cancelled(Message):
        pass

    def __init__(
        self,
        items: Sequence[SelectItem | str],
        *,
        current: str | None = None,
        enable_search: bool = True,
        search_placeholder: str = "Filter...",
        list_id: str | None = None,
        max_height: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(focusable=True, **kwargs)
        self._items = _normalize_select_items(items)
        self._filtered = list(self._items)
        self._current = current
        self._enable_search = enable_search
        self._search_placeholder = search_placeholder
        self._list_id = list_id or "select-list-view"
        self.max_height = max_height
        self.query = ""
        self._selected_index = 0
        if current is not None:
            for index, item in enumerate(self._items):
                if item.value == current:
                    self._selected_index = index
                    break

    @property
    def filtered_items(self) -> list[SelectItem]:
        return list(self._filtered)

    @property
    def selected_item(self) -> SelectItem | None:
        if 0 <= self._selected_index < len(self._filtered):
            return self._filtered[self._selected_index]
        return None

    @property
    def selected_index(self) -> int:
        return self._selected_index

    def move(self, direction: int) -> None:
        if not self._filtered:
            return
        self._selected_index = (self._selected_index + direction) % len(self._filtered)
        self.refresh()

    def handle_key(self, key: Key) -> bool:
        if key.name == "up":
            self.move(-1)
            return True
        if key.name == "down":
            self.move(1)
            return True
        if key.name == "enter":
            item = self.selected_item
            if item is not None:
                self.post_message(SelectList.Selected(item), "select_list")
            else:
                self.post_message(SelectList.Cancelled(), "select_list")
            return True
        if key.name == "escape":
            self.post_message(SelectList.Cancelled(), "select_list")
            return True
        if key.name == "backspace":
            self.query = self.query[:-1]
            self._apply_filter()
            return True
        if key.char is not None and key.char.isprintable() and self._enable_search:
            self.query += key.char
            self._apply_filter()
            return True
        return False

    def _apply_filter(self) -> None:
        self._filtered = _filter_items(self._items, self.query)
        self._selected_index = 0
        self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = []
        if self._enable_search:
            style = self.base_style
            if not self.query:
                style = (style or Style()) + Style(dim=True)
            lines.append(line_from_text(self.query or self._search_placeholder, width, style))
        visible = min(height - len(lines), len(self._filtered))
        for index in range(visible):
            item = self._filtered[index]
            marker = ">" if index == self._selected_index else " "
            if item.value == self._current and index != self._selected_index:
                marker = "•"
            label = f"{marker} {item.display_label}"
            if item.description:
                label += f"  {item.description}"
            style = self.base_style
            if index == self._selected_index:
                style = (style or Style()) + Style(reverse=True)
            lines.append(line_from_text(label, width, style))
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        return lines

    def content_size(self) -> tuple[int, int]:
        header = 1 if self._enable_search else 0
        width = max([len(item.display_label or "") for item in self._filtered] or [0])
        return (min(width + 2, 1000), min(header + len(self._filtered), self.max_height))


class SettingsList(SelectList):
    """设置项列表：Enter 循环取值。"""

    class Changed(Message):
        def __init__(self, item: SettingItem, value: str) -> None:
            super().__init__()
            self.item = item
            self.value = value

    class Activated(Message):
        def __init__(self, item: SettingItem) -> None:
            super().__init__()
            self.item = item

    def __init__(
        self,
        items: Sequence[SettingItem],
        *,
        enable_search: bool = False,
        list_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._setting_items = list(items)
        self._values = {item.id: item.current_value for item in self._setting_items}
        super().__init__(
            [item.label for item in self._setting_items],
            enable_search=enable_search,
            list_id=list_id,
            **kwargs,
        )

    def values(self) -> dict[str, str]:
        return dict(self._values)

    def _item_at(self, index: int) -> SettingItem | None:
        if 0 <= index < len(self._filtered):
            label = self._filtered[index].value
            for item in self._setting_items:
                if item.label == label:
                    return item
        return None

    def handle_key(self, key: Key) -> bool:
        if key.name == "enter":
            item = self._item_at(self._selected_index)
            if item is not None and item.values:
                values = item.values
                current = self._values.get(item.id, item.current_value)
                next_index = values.index(current) + 1 if current in values else 0
                next_value = values[next_index % len(values)]
                self._values[item.id] = next_value
                self.post_message(SettingsList.Changed(item, next_value), "settings_list")
                self.refresh()
            elif item is not None:
                self.post_message(SettingsList.Activated(item), "settings_list")
            return True
        return super().handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = []
        visible = min(height, len(self._filtered))
        for index in range(visible):
            item = self._filtered[index]
            setting = self._item_at(index)
            value = self._values.get(setting.id, setting.current_value) if setting else ""
            marker = ">" if index == self._selected_index else " "
            label = f"{marker} {item.display_label}  {value}"
            style = self.base_style
            if index == self._selected_index:
                style = (style or Style()) + Style(reverse=True)
            lines.append(line_from_text(label, width, style))
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        return lines


__all__ = [
    "Message",
    "Widget",
    "Container",
    "Vertical",
    "Horizontal",
    "VStack",
    "HStack",
    "Box",
    "Spacer",
    "Static",
    "Label",
    "Text",
    "Input",
    "Editor",
    "PiEditor",
    "PiEditorVim",
    "ScrollView",
    "Markdown",
    "Loader",
    "CancellableLoader",
    "AltScreenFlash",
    "SelectItem",
    "SettingItem",
    "SelectList",
    "SettingsList",
]
