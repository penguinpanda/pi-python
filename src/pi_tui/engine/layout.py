"""Flex 布局引擎：独立于组件树，产出 LayoutBox 树并合成为行缓冲。

对齐 TS packages/tui/src/layout.ts：
- 组件通过 layout_node() 声明 stack / scroll 节点，其余按叶子渲染；
- 尺寸分配采用 basis + grow + shrink + minSize + maxSize（类 CSS flexbox）；
- 同一帧内按 (组件, 宽, 高) 缓存渲染结果（框架级渲染缓存）；
- 合成阶段逐 box 裁剪并 patch 到行缓冲，ScrollView 在 box 层裁切。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rich.style import Style

from .cells import Line, blank_line
from .layout_node import (
    LayoutBox,
    LayoutFrame,
    LayoutNode,
    LayoutRect,
    StackLayoutEntry,
    get_layout_node,
)

MAX_LAYOUT = 2**31 - 1


@dataclass
class LayoutContext:
    viewport: LayoutRect
    render_cache: dict[int, dict[tuple[int, int], list[Line]]] = field(default_factory=dict)
    request_render: Callable[[], None] = lambda: None
    primary_scroll_view: Any = None


def _noop() -> None:
    return None


def intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect:
    x = max(a.x, b.x)
    y = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    return LayoutRect(x, y, max(0, right - x), max(0, bottom - y))


def render_cached(
    context: LayoutContext,
    component: Any,
    width: int,
    height: int,
) -> list[Line]:
    """框架级渲染缓存：同一帧内同一组件同尺寸只渲染一次。"""
    safe_width = max(1, int(width))
    safe_height = max(0, int(height))
    widths = context.render_cache.setdefault(id(component), {})
    key = (safe_width, safe_height)
    lines = widths.get(key)
    if lines is None:
        lines = list(component.render(safe_width, safe_height))
        widths[key] = lines
    return lines


def _natural_size(component: Any, width: int) -> tuple[int, int]:
    method = getattr(component, "natural_size", None)
    if callable(method):
        try:
            return method(width)
        except Exception:
            pass
    return component.content_size()


def natural_height(context: LayoutContext, component: Any, width: int) -> int:
    _width, height = _natural_size(component, width)
    return max(0, int(height))


def natural_width(context: LayoutContext, component: Any, width: int) -> int:
    width_value, _height = _natural_size(component, width)
    return max(0, int(width_value))


def with_parent(box: LayoutBox, parent: LayoutBox) -> LayoutBox:
    box.parent = parent
    return box


def translate_box(box: LayoutBox, delta_y: int) -> None:
    box.rect.y += delta_y
    for child in box.children:
        translate_box(child, delta_y)


def update_clips(box: LayoutBox, parent_clip: LayoutRect) -> None:
    box.clip = intersect(parent_clip, box.rect)
    for child in box.children:
        update_clips(child, box.clip)


def clamp_size(size: int, entry: StackLayoutEntry) -> int:
    minimum = max(0, int(entry.min_size))
    maximum = max(minimum, int(entry.max_size))
    return max(minimum, min(maximum, max(0, int(size))))


def _distribute(
    sizes: list[int],
    entries: list[StackLayoutEntry],
    amount: int,
    mode: str,
) -> None:
    remaining = amount
    while remaining > 0:
        candidates = [
            (index, entry)
            for index, entry in enumerate(entries)
            if (entry.grow > 0 and sizes[index] < entry.max_size)
            if mode == "grow"
        ]
        if mode == "shrink":
            candidates = [
                (index, entry)
                for index, entry in enumerate(entries)
                if entry.shrink > 0 and sizes[index] > entry.min_size
            ]
        if not candidates:
            return
        total_weight = sum(
            entry.grow if mode == "grow" else entry.shrink * max(1, sizes[index])
            for index, entry in candidates
        )
        if total_weight <= 0:
            return
        distributed = 0
        for index, entry in candidates:
            if remaining <= 0:
                break
            weight = entry.grow if mode == "grow" else entry.shrink * max(1, sizes[index])
            proposed = max(1, (remaining * weight) // total_weight)
            capacity = (
                entry.max_size - sizes[index] if mode == "grow" else sizes[index] - entry.min_size
            )
            delta = min(remaining, proposed, capacity)
            if delta <= 0:
                continue
            sizes[index] += delta if mode == "grow" else -delta
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(
    entries: list[StackLayoutEntry],
    intrinsic_sizes: list[int],
    available_size: int | None,
    gap: int,
) -> list[int]:
    """按 basis/grow/shrink/min/max 分配一行/一列尺寸。"""
    sizes = [
        clamp_size(
            intrinsic_sizes[index] if entry.basis in (None, "auto") else int(entry.basis),
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    if available_size is None:
        return sizes
    content_size = max(0, int(available_size) - max(0, len(entries) - 1) * int(gap))
    total = sum(sizes)
    if total < content_size:
        _distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        _distribute(sizes, entries, total - content_size, "shrink")
    return sizes


def visible_stack_entries(
    entries: list[StackLayoutEntry],
    viewport: LayoutRect,
) -> list[StackLayoutEntry]:
    return [entry for entry in entries if (entry.visible(viewport) if entry.visible else True)]


def _set_widget_rect(
    component: Any,
    rect: LayoutRect,
    content_origin: tuple[int, int] | None,
) -> None:
    """组件 rect 保持内容坐标系（scroll 内为内容行号，外部即屏幕坐标）。"""
    if content_origin is None:
        row, col = rect.y, rect.x
    else:
        row, col = rect.y - content_origin[0], rect.x - content_origin[1]
    setter = getattr(component, "rect", None)
    if isinstance(setter, tuple):
        try:
            component.rect = (row, col, rect.width, rect.height)
        except Exception:
            pass


def layout_component(
    context: LayoutContext,
    component: Any,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
    content_origin: tuple[int, int] | None = None,
) -> LayoutBox:
    safe_width = max(1, int(width))
    node: LayoutNode | None = get_layout_node(component)

    if node is None:
        natural = component.content_size()
        natural_h = max(0, int(natural[1]))
        allocated_height = max(0, int(height)) if height is not None else max(1, natural_h)
        lines = render_cached(context, component, safe_width, allocated_height)
        rect = LayoutRect(x, y, safe_width, allocated_height)
        box = LayoutBox(
            component=component,
            rect=rect,
            clip=intersect(clip, rect),
            lines=lines,
            layer=0,
        )
        _set_widget_rect(component, rect, content_origin)
        return box

    if node.type == "scroll":
        state = node.state
        content_width = int(state.get_content_width(safe_width))
        # 内容子树按内容坐标（y=0）布局并合成完整行缓冲，
        # 再统一平移到屏幕坐标并按视口 clip 裁切。
        content_clip = LayoutRect(clip.x, -MAX_LAYOUT, clip.width, MAX_LAYOUT * 2)
        child_box = layout_component(
            context,
            node.component,
            x,
            0,
            content_width,
            None,
            content_clip,
            content_origin=(0, x),
        )
        content_height = child_box.rect.height
        viewport_height = max(0, int(height)) if height is not None else content_height
        state.update_layout(content_height, viewport_height, context.request_render)
        content_lines = compose_box_lines(child_box, content_width)
        translate_box(child_box, y - int(state.scroll_top))
        if state.primary or context.primary_scroll_view is None:
            context.primary_scroll_view = state
        rect = LayoutRect(x, y, safe_width, viewport_height)
        child_clip = intersect(clip, rect)
        box = LayoutBox(
            component=component,
            rect=rect,
            clip=child_clip,
            children=[child_box],
            scroll_view=state,
            scroll_content_lines=content_lines,
            layer=0,
        )
        child_box.parent = box
        update_clips(child_box, child_clip)
        _set_widget_rect(component, rect, content_origin)
        return box

    entries = visible_stack_entries(list(node.entries), context.viewport)
    gap_total = max(0, len(entries) - 1) * int(node.gap)
    if node.type == "vstack":
        intrinsic_heights = [
            int(entry.basis)
            if isinstance(entry.basis, int)
            else natural_height(context, entry.component, safe_width)
            for entry in entries
        ]
        sizes = allocate_stack_sizes(entries, intrinsic_heights, height, int(node.gap))
        natural_total = sum(sizes) + gap_total
        allocated_height = natural_total if height is None else max(0, int(height))
        rect = LayoutRect(x, y, safe_width, allocated_height)
        box = LayoutBox(
            component=component,
            rect=rect,
            clip=intersect(clip, rect),
            layer=0,
        )
        _set_widget_rect(component, rect, content_origin)
        child_y = y
        for index, entry in enumerate(entries):
            box.children.append(
                with_parent(
                    layout_component(
                        context,
                        entry.component,
                        x,
                        child_y,
                        safe_width,
                        sizes[index],
                        box.clip,
                        content_origin,
                    ),
                    box,
                )
            )
            child_y += sizes[index] + int(node.gap)
        return box

    intrinsic_widths = [
        int(entry.basis)
        if isinstance(entry.basis, int)
        else natural_width(context, entry.component, safe_width)
        for entry in entries
    ]
    widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, int(node.gap))
    intrinsic_heights = [
        natural_height(context, entry.component, max(1, widths[index]))
        for index, entry in enumerate(entries)
    ]
    allocated_height = max(intrinsic_heights, default=0) if height is None else max(0, int(height))
    rect = LayoutRect(x, y, safe_width, allocated_height)
    box = LayoutBox(
        component=component,
        rect=rect,
        clip=intersect(clip, rect),
        layer=0,
    )
    _set_widget_rect(component, rect, content_origin)
    child_x = x
    for index, entry in enumerate(entries):
        natural_child_height = intrinsic_heights[index]
        child_height = (
            allocated_height
            if node.align == "stretch"
            else min(allocated_height, natural_child_height)
        )
        child_y = y
        if node.align == "center":
            child_y += (allocated_height - child_height) // 2
        elif node.align == "end":
            child_y += allocated_height - child_height
        child_width = widths[index]
        if child_width == 0:
            empty = LayoutBox(
                component=entry.component,
                rect=LayoutRect(child_x, child_y, 0, child_height),
                clip=LayoutRect(child_x, child_y, 0, 0),
                parent=box,
                layer=0,
            )
            box.children.append(empty)
            _set_widget_rect(
                entry.component,
                LayoutRect(child_x, child_y, 0, child_height),
                content_origin,
            )
        else:
            box.children.append(
                with_parent(
                    layout_component(
                        context,
                        entry.component,
                        child_x,
                        child_y,
                        child_width,
                        child_height,
                        box.clip,
                        content_origin,
                    ),
                    box,
                )
            )
        child_x += child_width + int(node.gap)
    return box


def _ensure_row(screen: list[Line], row: int, width: int, style: Style | None) -> None:
    while len(screen) <= row:
        screen.append(blank_line(width, style))


def _patch_clipped(
    target: Line,
    source: Line,
    target_x: int,
    source_x: int,
    count: int,
) -> None:
    for offset in range(count):
        target_col = target_x + offset
        source_col = source_x + offset
        if target_col < 0 or target_col >= len(target.cells):
            continue
        if source_col < 0 or source_col >= len(source.cells):
            continue
        target.cells[target_col] = source.cells[source_col]


def paint_box(box: LayoutBox, screen: list[Line], total_width: int) -> None:
    if box.lines is not None:
        offset = box.line_offset
        first_row = max(box.rect.y, box.clip.y, 0)
        last_row = min(
            box.rect.y + box.rect.height,
            box.clip.y + box.clip.height,
        )
        for row in range(first_row, last_row):
            _ensure_row(screen, row, total_width, None)
            source_line = box.lines[offset + row - box.rect.y]
            if source_line is None:
                continue
            target = screen[row]
            if source_line.passthrough and box.rect.x == 0 and box.rect.width >= total_width:
                target.passthrough = source_line.passthrough
            visible_x = max(box.rect.x, box.clip.x, 0)
            visible_right = min(
                box.rect.x + box.rect.width,
                box.clip.x + box.clip.width,
                total_width,
            )
            source_x = visible_x - box.rect.x
            _patch_clipped(
                target,
                source_line,
                visible_x,
                source_x,
                max(0, visible_right - visible_x),
            )
    for child in box.children:
        paint_box(child, screen, total_width)
    paint_scrollbar(box, screen, total_width)


def compose_box_lines(box: LayoutBox, width: int) -> list[Line]:
    screen: list[Line] = []
    paint_box(box, screen, width)
    return screen


def get_scrollbar_geometry(box: LayoutBox) -> dict[str, int] | None:
    state = box.scroll_view
    if (
        state is None
        or not state.is_scrollbar_visible
        or box.rect.width <= 0
        or box.rect.height <= 0
    ):
        return None
    content_height = (
        box.children[0].rect.height
        if box.children
        else (len(box.scroll_content_lines) if box.scroll_content_lines else 0)
    )
    track_height = box.rect.height
    min_thumb_height = min(2, track_height)
    thumb_height = max(
        min_thumb_height,
        min(track_height, round(track_height * track_height / max(1, content_height))),
    )
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_offset = (
        0 if max_scroll_top == 0 else round(int(state.scroll_top) / max_scroll_top * max_thumb_top)
    )
    column = box.rect.x + box.rect.width - 1
    if column < box.clip.x or column >= box.clip.x + box.clip.width:
        return None
    return {
        "column": column,
        "track_top": box.rect.y,
        "track_height": track_height,
        "thumb_top": box.rect.y + thumb_offset,
        "thumb_height": thumb_height,
        "max_scroll_top": max_scroll_top,
    }


def paint_scrollbar(box: LayoutBox, screen: list[Line], total_width: int) -> None:
    state = box.scroll_view
    geometry = get_scrollbar_geometry(box)
    if geometry is None or state is None:
        return
    column = geometry["column"]
    for offset in range(geometry["thumb_height"]):
        row = geometry["thumb_top"] + offset
        if row < box.clip.y or row >= box.clip.y + box.clip.height or row < 0 or row >= len(screen):
            continue
        line = screen[row]
        if column >= len(line.cells):
            continue
        cell = line.cells[column]
        if cell.char == " ":
            cell.char = "█"
        style = state.scrollbar_style(cell.style)
        cell.style = style


def contains_point(rect: LayoutRect, x: int, y: int) -> bool:
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def get_scroll_view_box(frame: LayoutFrame, scroll_view: Any) -> LayoutBox | None:
    def visit(box: LayoutBox) -> LayoutBox | None:
        if box.scroll_view is scroll_view:
            return box
        for child in box.children:
            match = visit(child)
            if match is not None:
                return match
        return None

    return visit(frame.root)


def get_scroll_views_at(frame: LayoutFrame, x: int, y: int) -> list[Any]:
    result: list[tuple[Any, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not contains_point(box.clip, x, y):
            return
        if box.scroll_view is not None and contains_point(box.rect, x, y):
            result.append((box.scroll_view, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda pair: pair[1], reverse=True)
    return [scroll_view for scroll_view, _depth in result]


def render_layout_frame(
    root: Any,
    width: int,
    height: int | None = None,
    request_render: Callable[[], None] | None = None,
) -> LayoutFrame:
    """构建一帧布局并合成行缓冲；height=None 时输出自然高度文档。"""
    safe_width = max(1, int(width))
    safe_height = max(1, int(height)) if height is not None else MAX_LAYOUT
    viewport = LayoutRect(0, 0, safe_width, safe_height)
    context = LayoutContext(
        viewport=viewport,
        request_render=request_render or _noop,
    )
    root_box = layout_component(
        context,
        root,
        0,
        0,
        safe_width,
        height,
        viewport,
    )
    base_style: Style | None = getattr(root, "base_style", None)
    lines: list[Line] = []
    if height is not None:
        lines = [blank_line(safe_width, base_style) for _ in range(safe_height)]
    paint_box(root_box, lines, safe_width)
    return LayoutFrame(
        root=root_box,
        width=safe_width,
        height=len(lines) if height is None else safe_height,
        lines=lines,
        primary_scroll_view=context.primary_scroll_view,
    )


def box_at(frame: LayoutFrame, x: int, y: int) -> LayoutBox | None:
    """屏幕坐标 → 包含该点的最深可见 box（含 scroll 内已平移的子树）。"""
    best: LayoutBox | None = None

    def visit(box: LayoutBox) -> None:
        nonlocal best
        if not contains_point(box.clip, x, y):
            return
        if contains_point(box.rect, x, y):
            best = box
        for child in box.children:
            visit(child)

    visit(frame.root)
    return best


__all__ = [
    "LayoutContext",
    "LayoutFrame",
    "LayoutRect",
    "allocate_stack_sizes",
    "box_at",
    "compose_box_lines",
    "get_scroll_view_box",
    "get_scroll_views_at",
    "layout_component",
    "paint_box",
    "render_cached",
    "render_layout_frame",
    "visible_stack_entries",
]
