"""布局引擎 / 渲染缓存 / LayoutBox 树 / Main-screen 增量差分测试。"""

from __future__ import annotations

from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.layout import (
    allocate_stack_sizes,
    box_at,
    render_layout_frame,
)
from pi_tui.engine.layout_node import StackLayoutEntry
from pi_tui.engine.widgets import (
    Container,
    HStack,
    ScrollView,
    Static,
    VStack,
    Vertical,
)


def test_flex_grow_distributes_extra_space() -> None:
    entries = [
        StackLayoutEntry(object(), basis=10, grow=1, shrink=0),
        StackLayoutEntry(object(), basis=5, grow=0, shrink=0),
    ]
    sizes = allocate_stack_sizes(entries, [10, 5], 20, 0)
    assert sizes == [15, 5]


def test_flex_shrink_respects_min_size() -> None:
    entries = [
        StackLayoutEntry(object(), basis=10, shrink=1, min_size=4),
        StackLayoutEntry(object(), basis=10, shrink=0),
    ]
    sizes = allocate_stack_sizes(entries, [10, 10], 10, 0)
    assert sizes == [4, 10]


def test_flex_clamps_basis_to_max_size() -> None:
    entries = [StackLayoutEntry(object(), basis=100, max_size=30)]
    sizes = allocate_stack_sizes(entries, [100], 200, 0)
    assert sizes == [30]


def test_layout_frame_builds_box_tree_and_sets_rects() -> None:
    root = VStack(gap=1)
    first = Static("a", height=2)
    second = Static("b", height=3)
    root.mount(first, basis=2, shrink=0)
    root.mount(second, basis=3, shrink=0)

    frame = render_layout_frame(root, 10, 6)

    assert frame.height == 6
    assert first.rect == (0, 0, 10, 2)
    assert second.rect == (3, 0, 10, 3)
    assert len(frame.root.children) == 2
    assert frame.root.children[0].component is first
    assert frame.root.children[1].component is second
    assert frame.lines[3].text().strip() == "b"
    hit = box_at(frame, 0, 4)
    assert hit is not None and hit.component is second


def test_hstack_allocates_widths_and_aligns() -> None:
    root = HStack(gap=1, align="center")
    first = Static("ab")
    second = Static("cd")
    root.mount(first, basis=2, shrink=0)
    root.mount(second, basis=2, shrink=0)

    frame = render_layout_frame(root, 10, 3)

    assert first.rect == (1, 0, 2, 1)
    assert second.rect == (1, 3, 2, 1)
    assert frame.lines[1].text().strip() == "ab cd"


class _Counting(Static):
    def __init__(self, text: str = "x") -> None:
        super().__init__(text)
        self.render_calls = 0

    def render(self, width: int, height: int):
        self.render_calls += 1
        return super().render(width, height)


def test_render_cache_renders_scroll_child_once_per_frame() -> None:
    body = Vertical()
    child = _Counting("x")
    body.mount(child, basis=1, shrink=0)
    view = ScrollView(body, scrollbar="always")
    root = Container(direction="vertical")
    root.mount(view, basis=0, grow=1)

    render_layout_frame(root, 20, 5)

    # 同一帧内滚动内容只渲染一次（框架级缓存去重）。
    assert child.render_calls == 1


def test_scroll_view_layout_translates_and_clips_content() -> None:
    body = Vertical()
    for index in range(10):
        body.mount(Static(f"line {index}", height=1))
    view = ScrollView(body)
    view.scroll_top = 3
    root = Container(direction="vertical")
    root.mount(view, basis=0, grow=1)

    frame = render_layout_frame(root, 10, 4)

    scroll_box = frame.root.children[0]
    assert scroll_box.scroll_view is view
    assert scroll_box.scroll_content_lines is not None
    assert len(scroll_box.scroll_content_lines) == 10
    assert view.rect == (0, 0, 10, 4)
    assert frame.lines[0].text().strip() == "line 3"
    assert frame.lines[3].text().strip() == "line 6"
    # 组件 rect 保持内容坐标（scroll_to_widget 需要），box 树为屏幕坐标。
    assert body.children[0].rect[0] == 0
    hit = box_at(frame, 0, 2)
    assert hit is not None and hit.component is body.children[5]


def test_regular_mode_rewrites_only_changed_lines() -> None:
    term = FakeTerminal(size=(20, 6))
    app = App(terminal=term, ui_mode="regular")
    first = Static("alpha", height=1)
    second = Static("beta", height=1)
    third = Static("gamma", height=1)
    app.screen.mount(first, basis=1, shrink=0)
    app.screen.mount(second, basis=1, shrink=0)
    app.screen.mount(third, basis=1, shrink=0)

    app._render_regular(force=True)
    assert "alpha" in term.output_text
    assert "beta" in term.output_text
    # 首帧清视口但保留 scrollback，文档从视口顶部开始。
    assert "\x1b[2J\x1b[H" in term.output_text
    term.reset_output()

    second.update("BETA")
    app._render_regular(False)
    delta = term.output_text

    assert "\x1b[2K" in delta
    assert "BETA" in delta
    assert "alpha" not in delta
    assert "gamma" not in delta
    assert "\x1b[2J" not in delta
    assert "\x1b[1A" in delta  # 增量光标上移一行


def test_regular_mode_clears_deleted_lines() -> None:
    term = FakeTerminal(size=(20, 6))
    app = App(terminal=term, ui_mode="regular")
    first = Static("alpha", height=1)
    second = Static("beta", height=1)
    third = Static("gamma", height=1)
    app.screen.mount(first, basis=1, shrink=0)
    app.screen.mount(second, basis=1, shrink=0)
    app.screen.mount(third, basis=1, shrink=0)
    app._render_regular(force=True)
    term.reset_output()

    app.screen.remove(third)
    app._render_regular(False)
    delta = term.output_text

    assert "gamma" not in delta
    assert "\x1b[2K" in delta
    assert "\x1b[2J" not in delta


def test_regular_mode_full_redraw_on_width_change() -> None:
    term = FakeTerminal(size=(20, 6))
    app = App(terminal=term, ui_mode="regular")
    app.screen.mount(Static("alpha", height=1), basis=1, shrink=0)
    app._render_regular(force=True)
    term.reset_output()

    term.set_size((30, 6))
    app._render_regular(False)

    assert "\x1b[2J\x1b[H\x1b[3J" in term.output_text
