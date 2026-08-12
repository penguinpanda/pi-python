"""TuiUIContext / ThemeFacade 单元测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui import DARK_THEME, Theme

from pi_coding_agent.modes.interactive.ui_context import TuiUIContext


class _StubEditor:
    text = ""

    def insert(self, value: str) -> None:
        self.text += value


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
        self._working_visible = True
        self._theme_call = None
        self._keybindings = object()

    def set_title(self, title: str) -> None:
        self.title = title

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

    def _set_overlay_renderer(self, key, renderer, options=None) -> None:
        self._overlay_calls.append((key, renderer, dict(options or {})))

    def _set_hidden_thinking_label(self, label=None) -> None:
        self._thinking_label = label

    def _set_working_message(self, text=None) -> None:
        self._working_message = text

    def _set_theme(self, theme=None) -> None:
        self._theme_call = theme

    async def _await_text_input(self, title: str, placeholder: str = "") -> str | None:
        return placeholder or title


def test_theme_facade_fg_bg():
    app = _StubApp()
    ctx = TuiUIContext(app)
    assert "\x1b[38;2;" in ctx.theme.fg("accent", "hi")
    assert "\x1b[48;2;" in ctx.theme.bg("error", "hi")
    assert ctx.theme.fg("missing", "hi") == "hi"
    assert ctx.theme.strikethrough("done") == "\x1b[9mdone\x1b[0m"


def test_get_theme_and_context_usage():

    class _Loader:
        def available(self):
            return ["dark", "light", "custom"]

        def load(self, name):
            return {"dark": "d", "light": "l", "custom": "c"}.get(name)

    class _SessionStats:
        def get_session_stats(self):
            return {"tokens": {"input": 10, "output": 5, "total": 15}, "cost": 0.5}

    app = _StubApp()
    app._theme_loader = _Loader()
    app._session = _SessionStats()
    ctx = TuiUIContext(app)
    assert ctx.get_theme("light") == "l"
    assert ctx.get_theme("missing") is None
    names = [item["name"] for item in ctx.get_all_themes()]
    assert names == ["dark", "light", "custom"]
    usage = ctx.get_context_usage()
    assert usage == {
        "inputTokens": 10,
        "outputTokens": 5,
        "totalTokens": 15,
        "costUsd": 0.5,
    }


def test_on_terminal_input_subscribe_unsubscribe():
    app = _StubApp()
    app._terminal_input_handlers = []
    app.add_terminal_input_handler = lambda handler: (  # type: ignore[method-assign]
        app._terminal_input_handlers.append(handler)
        or (lambda: app._terminal_input_handlers.remove(handler))
    )
    ctx = TuiUIContext(app)
    seen: list = []
    unsubscribe = ctx.on_terminal_input(lambda event: seen.append(event))
    assert unsubscribe is not None
    app._terminal_input_handlers[0](object())
    assert len(seen) == 1
    unsubscribe()
    assert app._terminal_input_handlers == []


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
    renderer_fn = lambda width, height: ["x"]  # noqa: E731
    ctx.set_overlay_renderer("ov3", renderer_fn, {"anchor": "center"})
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
        ("ov3", renderer_fn, {"anchor": "center"}),
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


@pytest.mark.asyncio
async def test_custom_resolves_via_done():
    app = _StubApp()
    ctx = TuiUIContext(app)

    class _Dialog:
        def __init__(self, done) -> None:
            self.done = done
            self.app = None

        def handle_key(self, _key) -> bool:
            return False

        def render(self, _width, _height):
            return []

        def content_size(self):
            return (0, 0)

    task = asyncio.create_task(ctx.custom(lambda tui, theme, keybindings, done: _Dialog(done)))
    await asyncio.sleep(0)
    app._screens[0][1]("custom-result")
    assert await task == "custom-result"


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
    ctx.set_overlay_renderer("ov3", object(), {"anchor": "center"})
    methods = [entry["method"] for entry in emitted]
    assert methods == [
        "setFooter",
        "setHeader",
        "setOverlay",
        "setOverlayComponent",
        "setOverlayRenderer",
    ]
    assert emitted[0]["text"] == "footer"
    assert emitted[1]["text"] == "header"
    assert emitted[2]["animate"] is True
    assert emitted[2]["duration"] == 0.3
    assert emitted[2]["border"] == "round"
    assert emitted[3]["componentType"] == "object"
    assert emitted[4]["method"] == "setOverlayRenderer"
    assert emitted[4]["componentType"] == "renderer"


def test_easter_egg_slash_commands_registered() -> None:
    from pi_coding_agent.modes.interactive.slash_commands import (
        SlashCommandRegistry,
        register_builtin_commands,
    )

    registry = SlashCommandRegistry()
    register_builtin_commands(registry)
    for name in ("debug", "arminsayshi", "dementedelves"):
        assert registry.get(name) is not None


def test_extension_working_and_editor_api() -> None:
    app = _StubApp()
    ctx = TuiUIContext(app)
    ctx.set_working_visible(False)
    assert app._working_visible is False
    ctx.set_working_indicator({"message": "Thinking...", "visible": True})
    assert app._working_message == "Thinking..."
    assert app._working_visible is True
    ctx.paste_to_editor("pasted")
    assert app._editor.text == "pasted"
    assert ctx.get_editor_text() == "pasted"


@pytest.mark.asyncio
async def test_extension_editor_dialog() -> None:
    app = _StubApp()
    ctx = TuiUIContext(app)
    result = await ctx.editor("Edit", "prefill")
    assert result == "prefill"
