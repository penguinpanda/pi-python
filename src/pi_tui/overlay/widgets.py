"""引擎 overlay 组件：OverlayLayer 与 OverlayWidget（无 Textual）。"""

from __future__ import annotations

from typing import Any

from pi_tui.engine.cells import blank_line
from pi_tui.engine.overlay_widget import OverlayWidget
from pi_tui.engine.widgets import Container


class OverlayLayer(Container):
    """所有 overlay 的绝对定位容器（引擎中为兼容占位，实际合成在 App._compose）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(direction="vertical", **kwargs)

    def render(self, width: int = 0, height: int = 0):
        if width <= 0 or height <= 0:
            return ""
        return [blank_line(width, self.base_style) for _ in range(height)]


__all__ = ["OverlayLayer", "OverlayWidget"]
