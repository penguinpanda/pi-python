"""Textual overlay 组件：OverlayLayer 容器与 OverlayWidget。"""

from __future__ import annotations

from typing import Any, cast

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from .model import OverlayOptions


class OverlayLayer(Widget):
    """所有 overlay 的绝对定位容器，叠加在 base 内容之上。"""

    DEFAULT_CSS = """
    OverlayLayer {
        position: absolute;
        width: 100%;
        height: 100%;
        layer: overlay;
    }
    """


class OverlayWidget(Static):
    """overlay 根组件：行文本模式 + 组件树模式（二选一）。

    - 行文本：渲染文本、边框与标题；capturing 时接收焦点。
    - 组件树：容纳任意 Textual 子组件（set_overlay_component），
      焦点落到子树内第一个可聚焦组件。
    """

    can_focus = True

    def __init__(
        self,
        key: str,
        lines: list[str],
        options: OverlayOptions,
        component: Any | None = None,
    ) -> None:
        self._key = key
        self._options = options
        self._component = component
        self._mode = "component" if component is not None else "lines"
        super().__init__("", id=f"pi-overlay-{key}")
        self.styles.layer = "overlay"
        self.styles.position = "absolute"
        if self._mode == "component":
            self.can_focus = False
        self._apply_style(options)
        if self._mode == "lines":
            self.update_content(lines)

    def compose(self) -> ComposeResult:
        if self._mode == "component" and self._component is not None:
            yield self._component

    def update_content(self, lines: list[str]) -> None:
        if self._mode == "component":
            self._component = None
            self._mode = "lines"
            self.can_focus = True
            self.refresh(recompose=True)
        self.update("\n".join(lines))

    def update_options(self, options: OverlayOptions) -> None:
        self._options = options
        self._apply_style(options)

    def set_component(self, component: Any) -> None:
        """切换为组件树模式（可复用同一根节点，避免重挂载）。"""
        self._component = component
        self._mode = "component"
        self.can_focus = False
        self.refresh(recompose=True)

    def focus(self, scroll_visible: bool = True) -> "OverlayWidget":
        """组件模式聚焦子树内第一个可聚焦组件；行文本模式聚焦根。"""
        if self._mode == "component":
            target = self._first_focusable(self._component)
            if target is not None:
                if target.is_attached:
                    target.focus(scroll_visible)
                else:
                    # 子组件尚未完成挂载：下个刷新周期再聚焦。
                    self.call_after_refresh(lambda: self.focus(scroll_visible))
                return self
        return super().focus(scroll_visible)

    @staticmethod
    def _first_focusable(node: Any) -> Any | None:
        if node is None:
            return None
        if getattr(node, "focusable", False):
            return node
        for child in getattr(node, "children", ()):
            found = OverlayWidget._first_focusable(child)
            if found is not None:
                return found
        return None

    def _apply_style(self, options: OverlayOptions) -> None:
        style = options.style
        if style.border:
            self.styles.border = cast(Any, (style.border, style.border_color or "white"))
        if style.title is not None:
            self.border_title = style.title

    def handle_event(self, event: Any) -> bool:
        """事件路由钩子：默认不消费任何事件（未来由子组件处理）。"""
        return False
