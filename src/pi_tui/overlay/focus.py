"""Overlay 焦点控制器。

只管理“哪个 overlay 拥有焦点权”以及恢复关系（pre_focus / blocked / resume），
不负责具体 Textual widget 的 focus()，也不做输入分发（由 manager/app 完成）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import OverlayEntry, RestoreMode


@dataclass
class FocusRestoreState:
    """inactive / active / blocked 三态焦点恢复状态。"""

    status: str = "inactive"
    overlay: OverlayEntry | None = None
    blocked_by: Any | None = None
    resume: RestoreMode | None = None
    resume_target: Any | None = None

    @classmethod
    def inactive(cls) -> "FocusRestoreState":
        return cls(status="inactive")

    @classmethod
    def active(cls, overlay: OverlayEntry) -> "FocusRestoreState":
        return cls(status="active", overlay=overlay)

    @classmethod
    def blocked(
        cls,
        overlay: OverlayEntry,
        blocked_by: Any,
        resume: RestoreMode,
        resume_target: Any | None = None,
    ) -> "FocusRestoreState":
        return cls(
            status="blocked",
            overlay=overlay,
            blocked_by=blocked_by,
            resume=resume,
            resume_target=resume_target,
        )


class OverlayFocusController:
    """overlay 焦点归属与恢复状态机。

    三态语义：inactive（无 overlay 焦点权）、active（= TS eligible：overlay
    拥有焦点权）、blocked（焦点暂借给基座组件，按 RestoreMode 恢复）。
    """

    def __init__(self) -> None:
        self.focused: OverlayEntry | None = None
        self.state: FocusRestoreState = FocusRestoreState.inactive()

    def show(self, entry: OverlayEntry, pre_focus: Any) -> bool:
        """overlay 显示时调用；返回是否应把焦点交给该 overlay。"""
        entry.pre_focus = pre_focus
        if entry.options.behavior.non_capturing:
            return False
        self.focused = entry
        self.state = FocusRestoreState.active(entry)
        return True

    def focus(self, entry: OverlayEntry) -> None:
        """overlay 明确获得焦点权。"""
        self.focused = entry
        self.state = FocusRestoreState.active(entry)

    def on_base_widget_focused(self, widget: Any) -> Any | None:
        """基座 widget 获得焦点；返回需要立即聚焦的恢复目标（若有）。"""
        focused = self.focused
        if focused is not None:
            # 焦点从 overlay 离开到基座：记录 blocked，等 blocked_by 失焦时恢复。
            self.focused = None
            self.state = FocusRestoreState.blocked(focused, widget, RestoreMode.OVERLAY)
            return None
        state = self.state
        if state.status == "blocked" and state.overlay is not None:
            if state.resume == RestoreMode.TARGET:
                target = state.resume_target
                self.state = FocusRestoreState.inactive()
                return target
            # 焦点在基座间移动：保持 blocked，更新 blocked_by。
            state.blocked_by = widget
        return None

    def release(self, entry: OverlayEntry, target: Any | None = None) -> Any | None:
        """handle.unfocus；返回应立即聚焦的 widget（None=无需立即变化）。"""
        if self.focused is not entry and self.state.overlay is not entry:
            return None
        if self.state.status == "blocked" and self.state.overlay is entry:
            if target is not None:
                self.state = FocusRestoreState.blocked(
                    entry,
                    self.state.blocked_by,
                    RestoreMode.TARGET,
                    target,
                )
                return None
            self.state = FocusRestoreState.inactive()
            return None
        self.state = FocusRestoreState.inactive()
        self.focused = None
        return target

    def on_hidden(self, entry: OverlayEntry) -> bool:
        """overlay 被隐藏 / 移除；返回 True 表示焦点需要重新分配。"""
        changed = False
        if self.focused is entry:
            self.focused = None
            changed = True
        if self.state.overlay is entry:
            self.state = FocusRestoreState.inactive()
            changed = True
        return changed

    def restore_on_input(self, current_focus: Any) -> Any | None:
        """输入前调用；若焦点应回到 overlay / 恢复目标，返回应聚焦的 widget。"""
        state = self.state
        if state.status == "active" and state.overlay is not None:
            if not _is_same_or_descendant(state.overlay.widget, current_focus):
                return state.overlay.widget
            return None
        if state.status == "blocked" and state.overlay is not None:
            if current_focus == state.blocked_by:
                return None
            if state.resume == RestoreMode.TARGET:
                target = state.resume_target
                self.state = FocusRestoreState.inactive()
                self.focused = None
                return target
            self.focused = state.overlay
            self.state = FocusRestoreState.active(state.overlay)
            return state.overlay.widget
        return None

    def retarget_pre_focus(self, removed: OverlayEntry, entries: list[OverlayEntry]) -> None:
        """移除 overlay 后，把其他 overlay 指向它的 pre_focus 链跳回。"""
        for entry in entries:
            if entry is not removed and entry.pre_focus is removed.widget:
                entry.pre_focus = removed.pre_focus


def _is_same_or_descendant(ancestor: Any, node: Any) -> bool:
    """node 是 ancestor 本身或 ancestor 的子树节点（沿 parent 链判断）。"""
    while node is not None:
        if node is ancestor:
            return True
        node = getattr(node, "parent", None)
    return False
