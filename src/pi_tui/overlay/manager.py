"""OverlayManager：生命周期、z-order、可见性与事件路由（不依赖 Textual）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .focus import OverlayFocusController
from .layout import OverlayRect, resolve_layout
from .model import OverlayEntry, OverlayHandle, OverlayOptions, parse_overlay_options


@dataclass
class OverlayHooks:
    """manager 与宿主 UI（Textual 等）之间的适配点。"""

    make_widget: Callable[[str, list[str], OverlayOptions], Any]
    update_widget: Callable[[Any, list[str], OverlayOptions], None]
    mount: Callable[[Any], None]
    remove: Callable[[Any], None]
    set_visible: Callable[[Any, bool], None]
    reposition: Callable[[Any, OverlayRect, OverlayOptions], None]
    focus: Callable[[Any], None]
    current_focus: Callable[[], Any | None]
    content_size: Callable[[Any], tuple[int, int]]
    bring_to_front: Callable[[Any], None] | None = None
    request_render: Callable[[], None] | None = None
    make_component_widget: Callable[[str, Any, OverlayOptions], Any] | None = None
    update_component: Callable[[Any, Any, OverlayOptions], None] | None = None


class OverlayManager:
    """管理所有 overlay：显示 / 隐藏 / 移除 / 置顶 / 布局 / 事件路由。"""

    def __init__(
        self,
        hooks: OverlayHooks,
        term_size: tuple[int, int] = (80, 24),
    ) -> None:
        self._hooks = hooks
        self._entries: dict[str, OverlayEntry] = {}
        self._order = 0
        self._term_size = term_size
        self._visible_cache: dict[str, bool] = {}
        self.controller = OverlayFocusController()

    @property
    def entries(self) -> dict[str, OverlayEntry]:
        return self._entries

    @property
    def term_size(self) -> tuple[int, int]:
        """当前终端尺寸（overlay 布局/渲染回调使用）。"""
        return self._term_size

    def get(self, key: str) -> OverlayEntry | None:
        return self._entries.get(key)

    def entry_for_widget(self, widget: Any) -> OverlayEntry | None:
        """按 widget（含子树节点）反查所属 overlay entry。"""
        return self._entry_for_widget(widget)

    def is_visible(self, entry: OverlayEntry) -> bool:
        if entry.hidden:
            return False
        visible = entry.options.behavior.visible
        if visible is None:
            return True
        return bool(visible(self._term_size[0], self._term_size[1]))

    def is_hidden(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is None or entry.hidden

    def is_focused(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and self.controller.focused is entry

    def topmost_visible(
        self,
        exclude: OverlayEntry | None = None,
        capturing_only: bool = True,
    ) -> OverlayEntry | None:
        top: OverlayEntry | None = None
        for entry in self._entries.values():
            if entry is exclude or not self.is_visible(entry):
                continue
            if capturing_only and entry.options.behavior.non_capturing:
                continue
            if top is None or entry.focus_order > top.focus_order:
                top = entry
        return top

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def show(
        self,
        key: str,
        lines: list[str],
        options: dict[str, Any] | OverlayOptions | None = None,
    ) -> OverlayHandle:
        """创建或更新 overlay；返回控制 handle。"""
        opts = parse_overlay_options(options)
        entry = self._entries.get(key)
        reorder = entry is not None
        if entry is None:
            widget = self._hooks.make_widget(key, list(lines), opts)
            entry = OverlayEntry(
                key=key,
                widget=widget,
                options=opts,
                focus_order=self._next_order(),
            )
            self._entries[key] = entry
            self._hooks.mount(widget)
        else:
            entry.hidden = False
            entry.kind = "lines"
            entry.options = opts
            entry.focus_order = self._next_order()
            self._hooks.update_widget(entry.widget, list(lines), opts)
        return self._finish_show(entry, reorder)

    def show_component(
        self,
        key: str,
        component: Any,
        options: dict[str, Any] | OverlayOptions | None = None,
    ) -> OverlayHandle:
        """创建或更新“组件树” overlay：容纳任意宿主 widget 子树。"""
        if self._hooks.make_component_widget is None or self._hooks.update_component is None:
            raise TypeError("OverlayHooks 未配置 make_component_widget / update_component")
        opts = parse_overlay_options(options)
        entry = self._entries.get(key)
        if entry is None:
            widget = self._hooks.make_component_widget(key, component, opts)
            entry = OverlayEntry(
                key=key,
                widget=widget,
                options=opts,
                kind="component",
                focus_order=self._next_order(),
            )
            self._entries[key] = entry
            self._hooks.mount(widget)
            return self._finish_show(entry, False)
        # 复用同一根 widget（Textual remove 是异步的，换根会撞同 id）。
        entry.hidden = False
        entry.kind = "component"
        entry.options = opts
        entry.focus_order = self._next_order()
        self._hooks.update_component(entry.widget, component, opts)
        return self._finish_show(entry, True)

    def _finish_show(self, entry: OverlayEntry, reorder: bool) -> OverlayHandle:
        """show / show_component 共用尾部：可见性、焦点、置顶。"""
        visible = self.is_visible(entry)
        self._visible_cache[entry.key] = visible
        self._hooks.set_visible(entry.widget, visible)

        take_focus = False
        if visible and not entry.options.behavior.non_capturing:
            take_focus = self.controller.show(entry, self._hooks.current_focus())
            # 新建 widget 已挂在 layer 末尾（天然置顶）；只有更新已有 overlay 才需要重排。
            if take_focus and reorder and self._hooks.bring_to_front is not None:
                self._hooks.bring_to_front(entry.widget)
            if take_focus:
                self._hooks.focus(entry.widget)

        self._request_render()
        return OverlayHandle(self, entry.key)

    def remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._visible_cache.pop(entry.key, None)
        self.controller.retarget_pre_focus(entry, list(self._entries.values()))
        changed = self.controller.on_hidden(entry)
        self._hooks.remove(entry.widget)
        if changed:
            self._focus_widget(self._fallback_widget(entry))
        self._request_render()

    def set_hidden(self, key: str, hidden: bool) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.hidden == hidden:
            return
        entry.hidden = hidden
        if hidden:
            self._visible_cache[entry.key] = False
            changed = self.controller.on_hidden(entry)
            self._hooks.set_visible(entry.widget, False)
            if changed:
                self._focus_widget(self._fallback_widget(entry))
        else:
            visible = self.is_visible(entry)
            self._visible_cache[entry.key] = visible
            if visible:
                self._hooks.set_visible(entry.widget, True)
                entry.focus_order = self._next_order()
                if self._hooks.bring_to_front is not None:
                    self._hooks.bring_to_front(entry.widget)
                if not entry.options.behavior.non_capturing:
                    self.controller.focus(entry)
                    self._hooks.focus(entry.widget)
                self._reposition(entry)
        self._request_render()

    def focus(self, key: str) -> None:
        """聚焦 overlay 并置顶。"""
        entry = self._entries.get(key)
        if entry is None or not self.is_visible(entry):
            return
        entry.focus_order = self._next_order()
        if self._hooks.bring_to_front is not None:
            self._hooks.bring_to_front(entry.widget)
        self.controller.focus(entry)
        self._hooks.focus(entry.widget)
        self._request_render()

    def ensure_focus(self, key: str) -> None:
        """overlay 仍持有焦点权时重新聚焦（不改变 z-order）。

        用于组件 overlay 的子组件完成挂载后，把焦点真正落到子树。
        blocked 且焦点被临时挪走（如重挂载中间态）时同样恢复。
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        if self.controller.focused is not entry:
            if self.controller.state.overlay is not entry:
                return
            self.controller.focus(entry)
        self._hooks.focus(entry.widget)

    def unfocus(self, key: str, target: Any | None = None) -> None:
        """释放 overlay 焦点；target 提供时作为显式恢复目标。"""
        entry = self._entries.get(key)
        if entry is None:
            return
        was_focused = self.controller.focused is entry
        decision = self.controller.release(entry, target)
        if decision is not None:
            self._focus_widget(decision)
        elif was_focused:
            self._focus_widget(self._fallback_widget(entry))
        self._request_render()

    # ------------------------------------------------------------------
    # 布局 / 尺寸
    # ------------------------------------------------------------------

    def reposition(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is None or not self.is_visible(entry):
            return
        self._reposition(entry)

    def _reposition(self, entry: OverlayEntry) -> None:
        content = self._hooks.content_size(entry.widget)
        rect = resolve_layout(entry.options.layout, content, self._term_size)
        self._hooks.reposition(entry.widget, rect, entry.options)

    def on_resize(self, size: tuple[int, int]) -> None:
        """终端尺寸变化：重算布局、按 visible 回调重定向焦点。"""
        self._term_size = (size[0], size[1])
        focused = self.controller.focused
        if focused is not None and not self.is_visible(focused):
            self.controller.on_hidden(focused)
            self._focus_widget(self._fallback_widget(focused))
        for entry in list(self._entries.values()):
            visible = self.is_visible(entry)
            prev = self._visible_cache.get(entry.key, visible)
            if visible != prev:
                self._visible_cache[entry.key] = visible
                self._hooks.set_visible(entry.widget, visible)
                if visible:
                    entry.focus_order = self._next_order()
                    if not entry.options.behavior.non_capturing:
                        self.controller.focus(entry)
                        self._hooks.focus(entry.widget)
            if visible:
                self._reposition(entry)
        self._request_render()

    # ------------------------------------------------------------------
    # 焦点同步 / 输入路由
    # ------------------------------------------------------------------

    def on_widget_focused(self, widget: Any) -> None:
        """宿主（Textual DescendantFocus）报告某 widget 获得焦点。"""
        entry = self._entry_for_widget(widget)
        if entry is not None:
            if self.controller.focused is not entry:
                entry.focus_order = self._next_order()
                if self._hooks.bring_to_front is not None:
                    self._hooks.bring_to_front(entry.widget)
                self.controller.focus(entry)
                self._request_render()
            return
        decision = self.controller.on_base_widget_focused(widget)
        if decision is not None:
            self._hooks.focus(decision)

    def route_input(self) -> None:
        """输入前调用：按恢复策略把焦点交回 overlay / 恢复目标。"""
        target = self.controller.restore_on_input(self._hooks.current_focus())
        if target is None:
            return
        entry = self._entry_for_widget(target)
        if entry is not None and not self.is_visible(entry):
            self.controller.on_hidden(entry)
            return
        self._hooks.focus(target)
        self._request_render()

    def handle_event(self, event: Any) -> bool:
        """事件路由：从最上层可见 capturing overlay 向下冒泡到基座。"""
        candidates = [
            e
            for e in self._entries.values()
            if self.is_visible(e) and not e.options.behavior.non_capturing
        ]
        candidates.sort(key=lambda e: e.focus_order, reverse=True)
        for entry in candidates:
            handler = getattr(entry.widget, "handle_event", None)
            if handler is not None and handler(event):
                return True
        return False

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _next_order(self) -> int:
        self._order += 1
        return self._order

    def _request_render(self) -> None:
        if self._hooks.request_render is not None:
            self._hooks.request_render()

    def _entry_for_widget(self, widget: Any) -> OverlayEntry | None:
        for entry in self._entries.values():
            node = widget
            while node is not None:
                if node is entry.widget:
                    return entry
                node = getattr(node, "parent", None)
        return None

    def _fallback_widget(self, entry: OverlayEntry) -> Any | None:
        top = self.topmost_visible(exclude=entry)
        if top is not None:
            return top.widget
        return entry.pre_focus

    def _focus_widget(self, widget: Any) -> None:
        if widget is None:
            return
        if isinstance(widget, OverlayEntry):
            widget = widget.widget
        entry = self._entry_for_widget(widget)
        if entry is not None:
            # 聚焦的是 overlay widget：同步焦点归属（宿主无 DescendantFocus 时也一致）。
            self.controller.focus(entry)
        self._hooks.focus(widget)
