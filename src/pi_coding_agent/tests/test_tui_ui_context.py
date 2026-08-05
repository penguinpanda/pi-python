"""TuiUIContext / ThemeFacade 单元测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui import DARK_THEME, Theme

from pi_coding_agent.modes.interactive.ui_context import TuiUIContext


class _StubEditor:
    text = ""


class _StubWidget:
    def __init__(self) -> None:
        self.text = ""

    def update(self, text: str) -> None:
        self.text = text


class _StubApp:
    def __init__(self) -> None:
        self.title = ""
        self._status: str | None = None
        self._editor = _StubEditor()
        self._footer = _StubWidget()
        self._header = _StubWidget()
        self._theme = Theme(name="dark", colors=dict(DARK_THEME))
        self._screens: list[tuple] = []
        self._replaced_editor = None
        self._widget_calls: list[tuple] = []
        self._overlay_calls: list[tuple] = []
        self._thinking_label = None
        self._working_message = None
        self._theme_call = None

    def push_screen(self, screen, callback=None) -> None:
        self._screens.append((screen, callback))

    def _set_status(self, text: str) -> None:
        self._status = text

    def _replace_editor(self, component) -> None:
        self._replaced_editor = component

    def _set_widget(self, key, lines, options=None) -> None:
        self._widget_calls.append((key, list(lines), dict(options or {})))

    def _set_overlay(self, key, lines, options=None) -> None:
        self._overlay_calls.append((key, list(lines), dict(options or {})))

    def _set_overlay_component(self, key, component, options=None) -> None:
        self._overlay_calls.append((key, component, dict(options or {})))

    def _set_hidden_thinking_label(self, label=None) -> None:
        self._thinking_label = label

    def _set_working_message(self, text=None) -> None:
        self._working_message = text

    def _set_theme(self, theme=None) -> None:
        self._theme_call = theme


def test_theme_facade_fg_bg():
    app = _StubApp()
    ctx = TuiUIContext(app)
    assert "\x1b[38;2;" in ctx.theme.fg("accent", "hi")
    assert "\x1b[48;2;" in ctx.theme.bg("error", "hi")
    assert ctx.theme.fg("missing", "hi") == "hi"


def test_set_status_editor_title():
    app = _StubApp()
    ctx = TuiUIContext(app)
    ctx.set_status("turn", "working")
    ctx.set_editor_text("prefilled")
    ctx.set_title("my pi")
    ctx.set_footer("git main • 42 tokens")
    ctx.set_header("custom header")
    editor_widget = object()
    ctx.set_editor_component(editor_widget)
    ctx.set_widget("w1", ["line1", "line2"], {"placement": "belowEditor"})
    ctx.set_overlay("ov1", ["overlay"], {"anchor": "center", "margin": 2})
    ctx.set_overlay_component("ov2", "panel", {"anchor": "top-left"})
    ctx.set_hidden_thinking_label("Pondering...")
    ctx.set_working_message("Working... (custom)")
    ctx.set_theme("light")
    assert app._status == "working"
    assert app._editor.text == "prefilled"
    assert app.title == "my pi"
    assert app._footer.text == "git main • 42 tokens"
    assert app._header.text == "custom header"
    assert app._replaced_editor is editor_widget
    assert app._widget_calls == [("w1", ["line1", "line2"], {"placement": "belowEditor"})]
    assert app._overlay_calls == [
        ("ov1", ["overlay"], {"anchor": "center", "margin": 2}),
        ("ov2", "panel", {"anchor": "top-left"}),
    ]
    assert app._thinking_label == "Pondering..."
    assert app._working_message == "Working... (custom)"
    assert app._theme_call == "light"


@pytest.mark.asyncio
async def test_select_confirm_input_resolve_callbacks():
    app = _StubApp()
    ctx = TuiUIContext(app)

    task = asyncio.create_task(ctx.select("T", ["a", "b"]))
    await asyncio.sleep(0)
    app._screens[0][1]("b")
    assert await task == "b"

    task = asyncio.create_task(ctx.confirm("T", "M"))
    await asyncio.sleep(0)
    app._screens[1][1]("Yes")
    assert await task is True

    task = asyncio.create_task(ctx.input("T"))
    await asyncio.sleep(0)
    app._screens[2][1]("value")
    assert await task == "value"


def test_rpc_ui_context_set_footer_header():
    from pi_coding_agent.rpc.rpc_mode import RpcUiContext

    emitted: list[dict] = []
    ctx = RpcUiContext(emit=emitted.append)
    ctx.set_footer("footer")
    ctx.set_header("header")
    ctx.set_overlay(
        "ov",
        ["x"],
        {"anchor": "center", "margin": 2, "animate": True, "duration": 0.3, "border": "round"},
    )
    ctx.set_overlay_component("ov2", object(), {"anchor": "top-left"})
    methods = [entry["method"] for entry in emitted]
    assert methods == ["setFooter", "setHeader", "setOverlay", "setOverlayComponent"]
    assert emitted[0]["text"] == "footer"
    assert emitted[1]["text"] == "header"
    assert emitted[2]["animate"] is True
    assert emitted[2]["duration"] == 0.3
    assert emitted[2]["border"] == "round"
    assert emitted[3]["componentType"] == "object"
