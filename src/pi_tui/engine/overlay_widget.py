"""OverlayWidget：overlay 根组件（行文本 / 组件树双模，无 App 依赖）。"""

from __future__ import annotations

from typing import Any

from ..overlay.model import OverlayOptions
from .cells import Cell, Line, blank_line, line_from_text
from .keys import Key
from .widgets import Widget


class OverlayWidget(Widget):
    """overlay 根组件：行文本模式 + 组件树模式。"""

    def __init__(
        self,
        key: str,
        lines: list[str],
        options: OverlayOptions,
        component: Any | None = None,
    ) -> None:
        super().__init__()
        self._key = key
        self._options = options
        self._lines = list(lines)
        self._component = component
        self._mode = "component" if component is not None else "lines"
        self._anim_task: Any | None = None

    @property
    def key(self) -> str:
        return self._key

    @property
    def options(self) -> OverlayOptions:
        return self._options

    def update_content(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._mode = "lines"
        self._component = None
        self.refresh()

    def update_options(self, options: OverlayOptions) -> None:
        self._options = options
        self.refresh()

    def set_component(self, component: Any) -> None:
        self._component = component
        self._mode = "component"
        self.refresh()

    def component(self) -> Any | None:
        return self._component

    def content_size(self) -> tuple[int, int]:
        if self._mode == "component" and self._component is not None:
            width, height = self._component.content_size()
            return (min(width + 2, 1000), min(height + 2, 200))
        width = max((len(line) for line in self._lines), default=0)
        return (min(width + 2, 1000), min(len(self._lines) + 2, 200))

    def render(self, width: int, height: int) -> list[Line]:
        if self._mode == "component" and self._component is not None:
            inner = self._component.render(max(0, width - 2), max(0, height - 2))
            return self._with_border(inner, width, height)
        inner = [line_from_text(line, max(0, width - 2)) for line in self._lines]
        return self._with_border(inner, width, height)

    def _with_border(self, inner: list[Line], width: int, height: int) -> list[Line]:
        style = self._options.style
        border = bool(style.border)
        border_color = style.border_color
        title = style.title
        base = self.base_style
        border_style = None
        if border:
            try:
                from rich.style import Style

                border_style = Style(color=border_color) if border_color else Style(color="white")
            except Exception:
                border_style = None
        if not border:
            result = inner[:height]
            while len(result) < height:
                result.append(blank_line(width, base))
            return result
        bordered: list[Line] = []
        title_text = title or ""
        for row in range(height):
            if row == 0:
                available = max(0, width - 4)
                shown = title_text[:available]
                fill = "─" * max(0, available - len(shown))
                bordered.append(line_from_text("╭─" + shown + fill + "╮", width, border_style))
            elif row == height - 1:
                bordered.append(
                    line_from_text("╰" + "─" * max(0, width - 2) + "╯", width, border_style)
                )
            else:
                line = blank_line(width, base)
                line.cells[0] = Cell("│", border_style)
                line.cells[-1] = Cell("│", border_style)
                inner_index = row - 1
                if 0 <= inner_index < len(inner):
                    line.patch(1, inner[inner_index])
                bordered.append(line)
        return bordered

    def handle_event(self, event: Any) -> bool:
        """事件路由：组件模式交给子树，行文本模式默认不消费。"""
        if self._mode == "component" and self._component is not None:
            if isinstance(event, Key) and self._component.handle_key(event):
                return True
        return False

    def first_focusable(self) -> Widget | None:
        if self._mode == "component" and self._component is not None:
            for widget in self._component.walk():
                if widget.focusable:
                    return widget
        return self

    def handle_key(self, key: Key) -> bool:
        return self.handle_event(key)


__all__ = ["OverlayWidget"]
