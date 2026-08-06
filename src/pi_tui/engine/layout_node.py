"""布局节点协议：组件通过 layout_node() 声明参与 Flex 布局的方式。

对齐 TS packages/tui/src/layout-node.ts：
- 普通组件没有布局节点，作为叶子直接按分配尺寸渲染；
- VStack/HStack 声明为 stack 节点（basis/grow/shrink/min/max）；
- ScrollView 声明为 scroll 节点（视口状态由组件自身持有）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from .cells import Line


@dataclass
class LayoutRect:
    """布局矩形（列 x、行 y、宽、高）；scroll 平移时原地修改 y。"""

    x: int
    y: int
    width: int
    height: int


@dataclass
class LayoutBox:
    """LayoutBox 树节点：独立于组件树，记录 rect / clip / 已渲染行。"""

    component: Any
    rect: LayoutRect
    clip: LayoutRect
    children: list["LayoutBox"] = field(default_factory=list)
    parent: "LayoutBox | None" = None
    lines: list[Line] | None = None
    line_offset: int = 0
    scroll_view: Any = None
    scroll_content_lines: list[Line] | None = None
    layer: int = 0


@dataclass(frozen=True)
class LayoutFrame:
    """一帧布局结果：根 box + 合成后的行缓冲。"""

    root: LayoutBox
    width: int
    height: int
    lines: list[Line]
    primary_scroll_view: Any = None


@dataclass
class StackLayoutEntry:
    """stack 子项选项（对齐 TS StackEntryOptions）。"""

    component: Any
    basis: int | str | None = None
    grow: int = 0
    shrink: int = 1
    min_size: int = 0
    max_size: int = 2**31 - 1
    visible: Callable[[LayoutRect], bool] | None = None


@dataclass(frozen=True)
class StackLayoutNode:
    type: Literal["vstack", "hstack"]
    entries: tuple[StackLayoutEntry, ...]
    gap: int = 0
    align: str = "stretch"


@dataclass(frozen=True)
class ScrollLayoutNode:
    type: Literal["scroll"] = "scroll"
    component: Any = None
    state: Any = None


LayoutNode = StackLayoutNode | ScrollLayoutNode


class LayoutComponent(Protocol):
    def layout_node(self) -> LayoutNode | None: ...


def get_layout_node(component: Any) -> LayoutNode | None:
    """读取组件的布局节点（无则 None，按叶子处理）。"""
    method = getattr(component, "layout_node", None)
    if callable(method):
        return method()
    return None


__all__ = [
    "LayoutBox",
    "LayoutComponent",
    "LayoutFrame",
    "LayoutNode",
    "LayoutRect",
    "ScrollLayoutNode",
    "StackLayoutEntry",
    "StackLayoutNode",
    "get_layout_node",
]
