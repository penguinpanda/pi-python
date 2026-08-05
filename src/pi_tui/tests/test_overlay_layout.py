"""Overlay 布局纯函数测试。"""

from __future__ import annotations

import random

from pi_tui.overlay import Margin, OverlayLayout, resolve_layout


def _layout(**kwargs) -> OverlayLayout:
    defaults = {
        "width": None,
        "min_width": None,
        "max_height": None,
        "row": None,
        "col": None,
        "anchor": "center",
        "margin": Margin(),
        "offset_x": 0,
        "offset_y": 0,
    }
    defaults.update(kwargs)
    return OverlayLayout(**defaults)


def test_anchor_positions() -> None:
    term = (80, 24)
    content = (10, 3)
    cases = {
        "top-left": (0, 0),
        "top-center": (0, 35),
        "top-right": (0, 70),
        "left-center": (10, 0),
        "center": (10, 35),
        "right-center": (10, 70),
        "bottom-left": (21, 0),
        "bottom-center": (21, 35),
        "bottom-right": (21, 70),
    }
    for anchor, (row, col) in cases.items():
        rect = resolve_layout(_layout(anchor=anchor), content, term)
        assert (rect.row, rect.col) == (row, col)


def test_margin_and_unknown_anchor_falls_back_to_center() -> None:
    layout = _layout(
        anchor="invalid",
        margin=Margin(top=1, right=2, bottom=3, left=4),
    )
    rect = resolve_layout(layout, (10, 3), (80, 24))
    # margin 可用区 74x20，内容 10x3 → center 在可用区内。
    assert rect.row == 1 + (20 - 3) // 2
    assert rect.col == 4 + (74 - 10) // 2


def test_percentage_width_row_col() -> None:
    layout = _layout(
        width="50%",
        row="25%",
        col="75%",
        margin=Margin(top=1, left=1),
    )
    rect = resolve_layout(layout, (0, 0), (81, 25))
    assert rect.width == 40  # 50% of (81-1)
    assert rect.row == 1 + (24 - 0) // 4
    assert rect.col == 1 + (80 - 40) * 3 // 4


def test_offset_and_clamp() -> None:
    layout = _layout(anchor="top-left", offset_x=100, offset_y=50)
    rect = resolve_layout(layout, (10, 3), (80, 24))
    assert rect.row == 24 - 3
    assert rect.col == 80 - 10


def test_max_height_and_min_width() -> None:
    layout = _layout(anchor="top-left", max_height=2, min_width=20)
    rect = resolve_layout(layout, (10, 5), (80, 24))
    assert rect.height == 2
    assert rect.width == 20


def test_random_layouts_stay_in_bounds() -> None:
    rng = random.Random(20260805)
    anchors = [
        "top-left",
        "top-center",
        "top-right",
        "left-center",
        "center",
        "right-center",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ]
    for _ in range(2000):
        term_w = rng.randint(5, 120)
        term_h = rng.randint(5, 60)
        content = (rng.randint(0, 60), rng.randint(0, 30))
        margin = Margin(
            top=rng.randint(0, 3),
            right=rng.randint(0, 3),
            bottom=rng.randint(0, 3),
            left=rng.randint(0, 3),
        )
        layout = OverlayLayout(
            width=rng.choice([None, rng.randint(0, 80), f"{rng.randint(0, 100)}%"]),
            min_width=rng.randint(0, 40),
            max_height=rng.choice([None, rng.randint(0, 30), f"{rng.randint(0, 100)}%"]),
            row=rng.choice([None, rng.randint(-20, 120), f"{rng.randint(0, 100)}%"]),
            col=rng.choice([None, rng.randint(-20, 120), f"{rng.randint(0, 100)}%"]),
            anchor=rng.choice(anchors),
            margin=margin,
            offset_x=rng.randint(-20, 40),
            offset_y=rng.randint(-20, 40),
        )
        rect = resolve_layout(layout, content, (term_w, term_h))
        avail_w = max(1, term_w - margin.left - margin.right)
        assert rect.width >= 0
        assert rect.height >= 0
        assert rect.width <= avail_w
        assert rect.row >= margin.top
        assert rect.col >= margin.left
        assert rect.row <= term_h
        assert rect.col <= term_w
        if margin.top + rect.height + margin.bottom <= term_h:
            assert rect.row + rect.height <= term_h - margin.bottom
        if margin.left + rect.width + margin.right <= term_w:
            assert rect.col + rect.width <= term_w - margin.right
