"""OverlayManager 生命周期 / z-order / 可见性 / 事件路由测试。"""

from __future__ import annotations

from pi_tui.overlay import OverlayHooks, OverlayManager


class _FakeWidget:
    def __init__(self, key: str) -> None:
        self.key = key
        self.display = True
        self.content = (10, 3)
        self.consume = False
        self.handled = None

    def handle_event(self, event) -> bool:
        if self.consume:
            self.handled = event
            return True
        return False


class _Env:
    def __init__(self) -> None:
        self.mounted: list[_FakeWidget] = []
        self.removed: list[_FakeWidget] = []
        self.repositioned: list[tuple[_FakeWidget, object]] = []
        self.focused: list[object] = []
        self.fronted: list[_FakeWidget] = []
        self.focus: object | None = None
        self.manager = OverlayManager(
            OverlayHooks(
                make_widget=lambda key, lines, options: _FakeWidget(key),
                update_widget=lambda widget, lines, options: None,
                make_component_widget=lambda key, component, options: component,
                update_component=lambda widget, component, options: None,
                mount=self.mounted.append,
                remove=self.removed.append,
                set_visible=lambda widget, visible: setattr(widget, "display", visible),
                reposition=lambda widget, rect, options: self.repositioned.append((widget, rect)),
                focus=self._focus,
                current_focus=lambda: self.focus,
                content_size=lambda widget: widget.content,
                bring_to_front=self.fronted.append,
            )
        )

    def _focus(self, widget) -> None:
        self.focused.append(widget)
        self.focus = widget


def test_show_mounts_and_focuses() -> None:
    env = _Env()
    handle = env.manager.show("a", ["hello"], {})
    assert len(env.mounted) == 1
    assert env.focused == [env.mounted[0]]
    assert handle.is_focused()


def test_non_capturing_keeps_focus() -> None:
    env = _Env()
    env.focus = "editor"
    env.manager.show("toast", ["x"], {"nonCapturing": True})
    assert env.focused == []
    assert env.manager.is_focused("toast") is False
    assert env.manager.topmost_visible(capturing_only=True) is None
    assert env.manager.topmost_visible(capturing_only=False) is not None


def test_remove_restores_pre_focus() -> None:
    env = _Env()
    env.focus = "editor"
    env.manager.show("a", ["x"], {})
    env.manager.remove("a")
    assert env.focused[-1] == "editor"


def test_stacked_overlays_restore_topmost_then_pre_focus() -> None:
    env = _Env()
    env.focus = "editor"
    env.manager.show("a", ["a"], {})
    env.manager.show("b", ["b"], {})
    entry_b = env.manager.get("b")
    assert entry_b is not None
    env.manager.remove("b")
    assert env.focused[-1] is env.manager.get("a").widget
    env.manager.remove("a")
    assert env.focused[-1] == "editor"


def test_set_hidden_moves_focus_and_back() -> None:
    env = _Env()
    env.focus = "editor"
    handle = env.manager.show("a", ["a"], {})
    assert handle is not None
    env.manager.set_hidden("a", True)
    assert handle.is_hidden()
    assert env.focused[-1] == "editor"
    env.manager.set_hidden("a", False)
    assert env.focused[-1] is env.manager.get("a").widget


def test_visible_callback_and_resize() -> None:
    env = _Env()
    visible = lambda width, height: width >= 100  # noqa: E731
    env.manager.show("v", ["x"], {"visible": visible})
    assert env.focused == []
    env.manager.on_resize((120, 30))
    assert env.manager.is_visible(env.manager.get("v"))
    assert env.focused[-1] is env.manager.get("v").widget
    assert len(env.repositioned) == 1
    env.manager.on_resize((80, 24))
    assert env.manager.is_focused("v") is False


def test_handle_event_routes_topmost_first_and_skips_non_capturing() -> None:
    env = _Env()
    env.manager.show("low", ["l"], {})
    env.manager.show("high", ["h"], {})
    env.manager.show("toast", ["t"], {"nonCapturing": True})
    low = env.manager.get("low").widget
    high = env.manager.get("high").widget
    toast = env.manager.get("toast").widget
    low.consume = True
    high.consume = True
    toast.consume = True
    assert env.manager.handle_event("key") is True
    assert high.handled == "key"
    assert low.handled is None
    high.consume = False
    assert env.manager.handle_event("key") is True
    assert low.handled == "key"
    # toast 虽是最上层，但 nonCapturing 不参与事件路由。
    low.consume = False
    toast.consume = True
    assert env.manager.handle_event("key") is False
    assert toast.handled is None


def test_focus_bumps_order_and_brings_to_front() -> None:
    env = _Env()
    env.manager.show("a", ["a"], {})
    env.manager.show("b", ["b"], {})
    low = env.manager.get("a").widget
    env.manager.focus("a")
    assert env.focused[-1] is low
    assert env.fronted[-1] is low
    assert env.manager.topmost_visible() is env.manager.get("a")


def test_show_component_mounts_and_focuses() -> None:
    env = _Env()
    env.focus = "editor"
    component = _FakeWidget("c")
    handle = env.manager.show_component("c", component, {})
    assert env.mounted[-1] is component
    assert env.focused[-1] is component
    assert handle.is_focused()
    assert env.manager.get("c").kind == "component"


def test_show_component_replaces_lines_overlay() -> None:
    env = _Env()
    env.focus = "editor"
    env.manager.show("a", ["lines"], {})
    lines_widget = env.manager.get("a").widget
    component = _FakeWidget("c")
    env.manager.show_component("a", component, {})
    assert env.manager.get("a").kind == "component"
    # 复用同一根 widget（避免 Textual 异步 remove 撞同 id）。
    assert env.manager.get("a").widget is lines_widget
    assert env.removed == []


def test_component_descendant_focus_syncs_entry() -> None:
    env = _Env()
    env.focus = "editor"
    component = _FakeWidget("c")
    env.manager.show_component("c", component, {})
    env.manager.on_widget_focused(component)
    assert env.manager.is_focused("c")


def test_route_input_keeps_component_focus() -> None:
    env = _Env()
    env.focus = "editor"
    component = _FakeWidget("c")
    env.manager.show_component("c", component, {})
    env.focus = component  # 模拟组件子树持有焦点。
    before = list(env.focused)
    env.manager.route_input()
    assert env.focused == before


def test_overlay_layer_renders_empty() -> None:
    """OverlayLayer 容器本身不渲染任何内容（防止默认描述文本遮挡 UI）。"""
    from pi_tui.overlay import OverlayLayer

    assert OverlayLayer().render() == ""
