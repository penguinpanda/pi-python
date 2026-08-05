"""Overlay 布局纯函数：锚点 / margin / 百分比 / offset / clamp。"""

from __future__ import annotations

from dataclasses import dataclass

from .model import OverlayLayout

_ANCHORS = frozenset(
    {
        "top-left",
        "top-center",
        "top-right",
        "left-center",
        "center",
        "right-center",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    }
)


@dataclass(frozen=True)
class OverlayRect:
    """解析后的绝对屏幕矩形（row/col 为左上角）。"""

    row: int
    col: int
    width: int
    height: int


def parse_size(value: int | str | None, reference: int) -> int | None:
    """把绝对值或百分比字符串（如 "50%"）解析为列/行数。"""
    if value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            try:
                percent = float(text[:-1])
            except ValueError:
                return None
            return max(0, int(reference * percent / 100))
    return None


def _normalize_anchor(anchor: str) -> str:
    return anchor if anchor in _ANCHORS else "center"


def resolve_anchor_row(anchor: str, height: int, avail_height: int, margin_top: int) -> int:
    anchor = _normalize_anchor(anchor)
    if anchor in ("bottom-left", "bottom-center", "bottom-right"):
        return margin_top + avail_height - height
    if anchor in ("left-center", "center", "right-center"):
        return margin_top + (avail_height - height) // 2
    return margin_top


def resolve_anchor_col(anchor: str, width: int, avail_width: int, margin_left: int) -> int:
    anchor = _normalize_anchor(anchor)
    if anchor in ("top-right", "right-center", "bottom-right"):
        return margin_left + avail_width - width
    if anchor in ("top-center", "center", "bottom-center"):
        return margin_left + (avail_width - width) // 2
    return margin_left


def resolve_layout(
    layout: OverlayLayout,
    content_size: tuple[int, int],
    terminal_size: tuple[int, int],
) -> OverlayRect:
    """解析 overlay 布局；结果保证在终端边界（含 margin）内。"""
    term_width, term_height = terminal_size
    content_width, content_height = content_size
    margin = layout.margin
    avail_width = max(1, term_width - margin.left - margin.right)
    avail_height = max(1, term_height - margin.top - margin.bottom)

    width = parse_size(layout.width, avail_width) or max(0, content_width)
    if layout.min_width is not None:
        width = max(width, layout.min_width)
    width = min(width, avail_width)

    height = max(0, content_height)
    max_height = parse_size(layout.max_height, avail_height)
    if max_height is not None:
        height = min(height, max_height)

    if layout.row is not None:
        if isinstance(layout.row, str):
            percent = parse_size(layout.row, max(0, avail_height - height)) or 0
            row = margin.top + percent
        else:
            row = layout.row
    else:
        row = resolve_anchor_row(layout.anchor, height, avail_height, margin.top)

    if layout.col is not None:
        if isinstance(layout.col, str):
            percent = parse_size(layout.col, max(0, avail_width - width)) or 0
            col = margin.left + percent
        else:
            col = layout.col
    else:
        col = resolve_anchor_col(layout.anchor, width, avail_width, margin.left)

    row += layout.offset_y
    col += layout.offset_x

    row = max(margin.top, min(row, term_height - margin.bottom - height))
    col = max(margin.left, min(col, term_width - margin.right - width))
    return OverlayRect(row=row, col=col, width=width, height=height)
