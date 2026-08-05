"""OverlayFocusController 状态机测试。"""

from __future__ import annotations

from pi_tui.overlay import (
    OverlayBehavior,
    OverlayEntry,
    OverlayFocusController,
    OverlayOptions,
    RestoreMode,
)


class _Widget:
    def __init__(self, name: str) -> None:
        self.name = name


class _Child:
    def __init__(self, parent) -> None:
        self.parent = parent


def _entry(name: str, non_capturing: bool = False) -> OverlayEntry:
    return OverlayEntry(
        key=name,
        widget=_Widget(name),
        options=OverlayOptions(behavior=OverlayBehavior(non_capturing=non_capturing)),
    )


def test_show_capturing_takes_focus() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    assert controller.show(entry, "editor") is True
    assert controller.focused is entry
    assert controller.state.status == "active"
    assert entry.pre_focus == "editor"


def test_show_non_capturing_keeps_focus() -> None:
    controller = OverlayFocusController()
    entry = _entry("toast", non_capturing=True)
    assert controller.show(entry, "editor") is False
    assert controller.focused is None
    assert controller.state.status == "inactive"


def test_blur_to_base_blocks_and_resumes_on_input() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    assert controller.on_base_widget_focused("editor") is None
    assert controller.state.status == "blocked"
    assert controller.state.overlay is entry
    assert controller.state.blocked_by == "editor"
    assert controller.state.resume == RestoreMode.OVERLAY
    # 输入仍发生在 blocked_by 上：不恢复。
    assert controller.restore_on_input("editor") is None
    # 焦点在基座间移动：保持 blocked，更新 blocked_by。
    assert controller.on_base_widget_focused("other") is None
    assert controller.state.blocked_by == "other"
    # 输入仍由当前 blocked_by 处理，不自动恢复。
    assert controller.restore_on_input("other") is None
    # 焦点回到 overlay 子树：恢复 overlay 焦点权。
    controller.focus(entry)
    assert controller.focused is entry
    assert controller.state.status == "active"


def test_release_active_focus_returns_target() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    assert controller.release(entry, "target-widget") == "target-widget"
    assert controller.state.status == "inactive"
    assert controller.focused is None


def test_release_blocked_with_target_defers_resume() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    controller.on_base_widget_focused("editor")
    assert controller.release(entry, target="target-widget") is None
    assert controller.state.status == "blocked"
    assert controller.state.resume == RestoreMode.TARGET
    # blocked_by 失去焦点 → 跳到显式目标。
    assert controller.on_base_widget_focused("other") == "target-widget"
    assert controller.state.status == "inactive"


def test_hide_focused_clears_state() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    assert controller.on_hidden(entry) is True
    assert controller.focused is None
    assert controller.state.status == "inactive"


def test_retarget_pre_focus_chain() -> None:
    controller = OverlayFocusController()
    first = _entry("a")
    second = _entry("b")
    controller.show(first, "editor")
    controller.show(second, first.widget)
    assert second.pre_focus is first.widget
    controller.retarget_pre_focus(first, [second])
    assert second.pre_focus == "editor"


def test_active_restore_on_input_when_focus_elsewhere() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    assert controller.restore_on_input("editor") is entry.widget
    assert controller.state.status == "active"
    assert controller.focused is entry


def test_active_no_restore_when_focus_inside_component_tree() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    child = _Child(entry.widget)
    assert controller.restore_on_input(child) is None


def test_blocked_resume_when_focus_cleared() -> None:
    controller = OverlayFocusController()
    entry = _entry("a")
    controller.show(entry, "editor")
    controller.on_base_widget_focused("editor")
    assert controller.restore_on_input(None) is entry.widget
    assert controller.state.status == "active"
    assert controller.focused is entry
