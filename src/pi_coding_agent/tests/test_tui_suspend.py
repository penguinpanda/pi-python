"""Suspend（Ctrl+Z）测试。"""

from __future__ import annotations

import asyncio
import signal
import sys

import pytest

from pi_tui.keybindings import DEFAULT_APP_KEYBINDINGS


def test_app_suspend_keybinding_present() -> None:
    binding = DEFAULT_APP_KEYBINDINGS["app.suspend"]
    assert binding.action == "suspend"
    if sys.platform == "win32":
        assert binding.key == ""
    else:
        assert binding.key == "ctrl+z"


def test_action_suspend_windows_unsupported(monkeypatch) -> None:
    import pi_coding_agent.modes.interactive.app as app_mod

    monkeypatch.setattr(app_mod.sys, "platform", "win32")

    class _App:
        def _notify(self, message):
            self.message = message

    app = _App()
    app_mod.PiTuiApp.action_suspend(app)
    assert "not supported on Windows" in app.message


@pytest.mark.asyncio
async def test_action_suspend_posix_flow(monkeypatch) -> None:
    """POSIX：exit → SIGTSTP 挂起 → SIGCONT 恢复重绘。"""
    import pi_coding_agent.modes.interactive.app as app_mod

    captured: dict = {}

    class _Terminal:
        async def exit(self, alt_screen=False):
            captured["exit"] = alt_screen

        async def enter(self, alt_screen=True):
            captured["enter"] = alt_screen

    class _App:
        ui_mode = "fullscreen"
        terminal = _Terminal()
        rendered = 0

        def _notify(self, message):
            captured["notify"] = message

        def _run_task(self, task):
            captured["task"] = task

        def request_render(self):
            self.rendered += 1

        async def _restore_after_suspend(self):
            await self.terminal.enter(alt_screen=True)
            self.rendered += 1

    app = _App()

    def fake_kill(pid, sig):
        captured["killed"] = sig
        assert sig == signal.SIGTSTP

    monkeypatch.setattr(app_mod.os, "kill", fake_kill)
    fake_handlers: dict = {}
    real_get_loop = asyncio.get_running_loop

    class _Loop:
        def add_signal_handler(self, s, cb):
            fake_handlers[s] = cb

        def remove_signal_handler(self, s):
            fake_handlers.pop(s, None)

        def create_task(self, t):
            return real_get_loop().create_task(t)

    monkeypatch.setattr(app_mod.asyncio, "get_running_loop", lambda: _Loop())
    # 拦截 signal.signal 避免改真实进程信号处理
    monkeypatch.setattr(app_mod.signal, "signal", lambda s, h: None)

    app_mod.PiTuiApp.action_suspend(app)
    suspend_task = captured["task"]
    await suspend_task  # exit → handlers → kill（桩）

    assert captured.get("exit") is True
    assert captured.get("killed") == signal.SIGTSTP

    # 模拟 SIGCONT：触发恢复
    fake_handlers[signal.SIGCONT]()
    for _ in range(50):
        if app.rendered >= 1:
            break
        await asyncio.sleep(0.01)
    assert captured.get("enter") is True
    assert app.rendered == 1
