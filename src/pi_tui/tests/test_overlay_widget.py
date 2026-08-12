"""OverlayWidget 行文本 / 组件树双模式测试。"""

from __future__ import annotations

from pi_tui.engine.keys import Key
from pi_tui.engine.overlay_widget import OverlayWidget
from pi_tui.engine.widgets import Input
from pi_tui.overlay.model import OverlayOptions, OverlayStyle


def _options(**style_kwargs) -> OverlayOptions:
    return OverlayOptions(style=OverlayStyle(**style_kwargs))


def test_lines_mode_without_border() -> None:
    widget = OverlayWidget("k", ["a", "bb"], _options())
    assert widget.key == "k"
    assert widget.content_size() == (4, 4)
    lines = widget.render(10, 3)
    assert lines[0].text().strip() == "a"
    assert lines[1].text().strip() == "bb"


def test_lines_mode_with_border_and_title() -> None:
    widget = OverlayWidget("k", ["body"], _options(border=True, title="Title"))
    lines = widget.render(10, 4)
    assert lines[0].text().startswith("╭─Title")
    assert lines[-1].text().startswith("╰")
    assert widget.handle_key(Key(name="escape")) is False


class _FakeComponent:
    def __init__(self) -> None:
        self._input = Input(value="x")

    def content_size(self) -> tuple[int, int]:
        return (5, 3)

    def render(self, width, height):
        return self._input.render(width, height)

    def walk(self):
        return [self._input]

    def handle_key(self, key: Key) -> bool:
        return key.name == "enter"


def test_component_mode() -> None:
    component = _FakeComponent()
    widget = OverlayWidget("k", [], _options(border=True), component=component)
    assert widget.component() is component
    assert widget.content_size() == (7, 5)
    lines = widget.render(10, 6)
    assert lines[0].text().startswith("╭")
    assert widget.handle_key(Key(name="escape")) is False
    assert widget.handle_key(Key(name="enter")) is True
    assert widget.first_focusable() is component._input


def test_update_content_and_options() -> None:
    widget = OverlayWidget("k", ["old"], _options())
    component = _FakeComponent()
    widget.set_component(component)
    assert widget.component() is component
    widget.update_content(["new"])
    assert widget.component() is None
    assert widget.content_size()[1] == 3
    widget.update_options(_options(border=True))
    assert widget.options.style.border is True
