"""TUI 首次设置组件（对齐 TS FirstTimeSetupComponent）。"""

from __future__ import annotations

from typing import Any, Callable

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import Widget


THEME_OPTIONS = (("dark", "Dark"), ("light", "Light"))
ANALYTICS_OPTIONS = ((True, "Share anonymous usage data"), (False, "Don't share"))


class FirstTimeSetupComponent(Widget):
    """两步对话框：主题选择，然后 analytics opt-in。"""

    def __init__(
        self,
        detected_theme: str,
        on_theme_preview: Callable[[str], None],
        on_submit: Callable[[str, bool], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self._step = "theme"
        self._detected_theme = detected_theme
        self._on_theme_preview = on_theme_preview
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._theme_index = 0
        for index, (value, _label) in enumerate(THEME_OPTIONS):
            if value == detected_theme:
                self._theme_index = index
                break
        self._analytics_index = 0

    def _move(self, delta: int) -> None:
        if self._step == "theme":
            next_index = max(0, min(len(THEME_OPTIONS) - 1, self._theme_index + delta))
            if next_index != self._theme_index:
                self._theme_index = next_index
                self._on_theme_preview(THEME_OPTIONS[next_index][0])
        else:
            self._analytics_index = max(
                0,
                min(len(ANALYTICS_OPTIONS) - 1, self._analytics_index + delta),
            )
        self.refresh()

    def handle_key(self, key: Key) -> bool:
        if key.name in ("up", "k"):
            self._move(-1)
            return True
        if key.name in ("down", "j"):
            self._move(1)
            return True
        if key.name in ("enter", "return"):
            if self._step == "theme":
                self._step = "analytics"
                self.refresh()
            else:
                self._on_submit(
                    THEME_OPTIONS[self._theme_index][0],
                    ANALYTICS_OPTIONS[self._analytics_index][0],
                )
            return True
        if key.name == "escape":
            self._on_cancel()
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = [
            line_from_text("Welcome to pi, the minimal coding agent.", width),
            blank_line(width),
        ]
        if self._step == "theme":
            lines.append(line_from_text("Pick a theme.", width))
            lines.append(
                line_from_text(f"Detected system appearance: {self._detected_theme}", width)
            )
            lines.append(blank_line(width))
            options: tuple[tuple[Any, str], ...] = THEME_OPTIONS
            selected = self._theme_index
        else:
            lines.append(line_from_text("Opt-in to anonymous usage data sharing?", width))
            lines.append(blank_line(width))
            options = ANALYTICS_OPTIONS
            selected = self._analytics_index
        for index, (_value, label) in enumerate(options):
            marker = "→" if index == selected else " "
            lines.append(line_from_text(f"{marker} {label}", width))
        lines.append(blank_line(width))
        lines.append(
            line_from_text(
                "Up/Down navigate, Enter "
                + ("continue" if self._step == "theme" else "finish")
                + ", Esc skip",
                width,
            )
        )
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (50, 10)
