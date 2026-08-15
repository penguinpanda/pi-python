"""App.register_action_handler 分发测试(扩展快捷键等动态绑定)。"""

from __future__ import annotations

from pi_tui.engine.app import App
from pi_tui.engine.keys import Key
from pi_tui.engine.terminal import FakeTerminal
from pi_tui.keybindings import Keybinding, KeybindingsManager


def test_action_handler_dispatch() -> None:
    kb = KeybindingsManager()
    kb.register(
        Keybinding(
            key="ctrl+alt+p",
            action_id="ext.ctrl_alt_p",
            action="ext_ctrl_alt_p",
            description="Plan",
        )
    )
    app = App(terminal=FakeTerminal(size=(80, 24)), keybindings=kb)
    calls: list[str] = []
    app.register_action_handler("ext.ctrl_alt_p", lambda: calls.append("ran"))
    app._dispatch_binding(Key(name="ctrl+alt+p"))
    assert calls == ["ran"]


def test_action_handler_missing_is_silent() -> None:
    kb = KeybindingsManager()
    kb.register(
        Keybinding(
            key="ctrl+alt+p",
            action_id="ext.ctrl_alt_p",
            action="ext_ctrl_alt_p",
            description="Plan",
        )
    )
    app = App(terminal=FakeTerminal(size=(80, 24)), keybindings=kb)
    # 未注册 handler 时不抛异常(保持原静默语义)。
    app._dispatch_binding(Key(name="ctrl+alt+p"))
