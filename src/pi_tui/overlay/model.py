"""Overlay 数据模型（无 UI 框架依赖，可单测）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from .manager import OverlayManager

VisiblePredicate = Callable[[int, int], bool]
MarginValue = int | dict[str, int] | None


class RestoreMode(str, Enum):
    """blocked 状态下的焦点恢复方式。"""

    OVERLAY = "overlay"
    TARGET = "target"


@dataclass(frozen=True)
class Margin:
    """四边留白。"""

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0

    @classmethod
    def from_value(cls, value: MarginValue) -> "Margin":
        if value is None:
            return cls()
        if isinstance(value, int):
            value = max(0, value)
            return cls(top=value, right=value, bottom=value, left=value)
        return cls(
            top=max(0, int(value.get("top", 0))),
            right=max(0, int(value.get("right", 0))),
            bottom=max(0, int(value.get("bottom", 0))),
            left=max(0, int(value.get("left", 0))),
        )


@dataclass
class OverlayLayout:
    """定位与尺寸：宽高 / 锚点 / margin / 百分比坐标 / offset。"""

    width: int | str | None = None
    min_width: int | None = None
    max_height: int | str | None = None
    row: int | str | None = None
    col: int | str | None = None
    anchor: str = "center"
    margin: Margin = field(default_factory=Margin)
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class OverlayStyle:
    """渲染样式。"""

    border: str | None = None
    border_color: str | None = None
    title: str | None = None


@dataclass
class OverlayBehavior:
    """行为选项：是否抢焦点、可见性回调、动画。"""

    non_capturing: bool = False
    visible: VisiblePredicate | None = None
    animate: bool = False
    duration: float = 0.5


@dataclass
class OverlayOptions:
    """Overlay 完整选项（layout / style / behavior 三类职责分离）。"""

    layout: OverlayLayout = field(default_factory=OverlayLayout)
    style: OverlayStyle = field(default_factory=OverlayStyle)
    behavior: OverlayBehavior = field(default_factory=OverlayBehavior)


@dataclass
class OverlayEntry:
    """单个 overlay 的运行时状态。"""

    key: str
    widget: Any
    options: OverlayOptions
    pre_focus: Any | None = None
    hidden: bool = False
    focus_order: int = 0
    kind: Literal["lines", "component"] = "lines"


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value or fallback)
    except (TypeError, ValueError):
        return fallback


def parse_overlay_options(options: dict[str, Any] | OverlayOptions | None) -> OverlayOptions:
    """把 set_overlay(key, lines, options) 的 dict 归一成 OverlayOptions。"""
    if isinstance(options, OverlayOptions):
        return options
    raw = dict(options or {})
    layout = OverlayLayout(
        width=raw.get("width"),
        min_width=raw.get("min_width") or raw.get("minWidth"),
        max_height=raw.get("max_height") or raw.get("maxHeight"),
        row=raw.get("row"),
        col=raw.get("col"),
        anchor=str(raw.get("anchor", "center")),
        margin=Margin.from_value(raw.get("margin")),
        offset_x=_as_int(
            raw.get("offset_x") if raw.get("offset_x") is not None else raw.get("offsetX")
        ),
        offset_y=_as_int(
            raw.get("offset_y") if raw.get("offset_y") is not None else raw.get("offsetY")
        ),
    )
    style = OverlayStyle(
        border=raw.get("border"),
        border_color=raw.get("border_color") or raw.get("borderColor"),
        title=raw.get("title"),
    )
    behavior = OverlayBehavior(
        non_capturing=bool(raw.get("non_capturing") or raw.get("nonCapturing")),
        visible=raw.get("visible"),
        animate=bool(raw.get("animate")),
        duration=float(raw.get("duration", 0.5)),
    )
    return OverlayOptions(layout=layout, style=style, behavior=behavior)


class OverlayHandle:
    """控制单个 overlay 的生命周期（对齐 TS OverlayHandle）。"""

    def __init__(self, manager: "OverlayManager", key: str) -> None:
        self._manager = manager
        self._key = key

    def hide(self) -> None:
        """永久移除 overlay。"""
        self._manager.remove(self._key)

    def set_hidden(self, hidden: bool) -> None:
        """临时隐藏 / 显示 overlay。"""
        self._manager.set_hidden(self._key, hidden)

    def is_hidden(self) -> bool:
        return self._manager.is_hidden(self._key)

    def focus(self) -> None:
        """聚焦该 overlay 并置顶。"""
        self._manager.focus(self._key)

    def unfocus(self, target: Any | None = None) -> None:
        """释放焦点（可指定显式目标）。"""
        self._manager.unfocus(self._key, target=target)

    def is_focused(self) -> bool:
        return self._manager.is_focused(self._key)
