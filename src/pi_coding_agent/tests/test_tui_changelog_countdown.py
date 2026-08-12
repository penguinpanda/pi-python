"""对话框倒计时与启动 changelog 通知测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent.modes.interactive.ui_context import TuiUIContext


class _Dialog:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def update_title(self, title: str) -> None:
        self.titles.append(title)


class _StubApp:
    def __init__(self) -> None:
        self.title = ""

    def push_screen(self, screen, callback=None) -> None:
        pass


@pytest.mark.asyncio
async def test_select_countdown_updates_title() -> None:
    app = _StubApp()
    ctx = TuiUIContext(app)
    dialog = _Dialog()
    task = ctx._start_countdown(dialog, "Pick", 2.0)
    await asyncio.sleep(0.1)
    assert dialog.titles
    assert "auto-cancel" in dialog.titles[-1]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_start_countdown_none_for_no_timeout() -> None:
    ctx = TuiUIContext(_StubApp())
    assert ctx._start_countdown(_Dialog(), "Pick", None) is None


def test_startup_changelog_first_run_records_version(monkeypatch, tmp_path) -> None:
    """fresh install（无 lastChangelogVersion）只记录版本、不显示。"""
    import pi_coding_agent.modes.interactive.app as app_mod

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## [0.2.0]\n\n- item\n", encoding="utf-8")

    class _Messages:
        def get_messages(self):
            return []

    class _Settings:
        def __init__(self):
            self.recorded = None

        def get_last_changelog_version(self):
            return None

        def set_last_changelog_version(self, version):
            self.recorded = version

    class _Chat:
        def __init__(self):
            self.messages = []

        def add_message_agent(self, message):
            self.messages.append(message)

    class _App:
        def __init__(self):
            self._session = _Messages()
            self._settings_manager = _Settings()
            self._chat = _Chat()

    import pi_coding_agent._config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_changelog_path",
        lambda: changelog,
    )
    app = _App()
    app_mod.PiTuiApp._show_startup_changelog_if_needed(app)
    assert app._settings_manager.recorded == "0.2.0"
    assert app._chat.messages == []


def test_startup_changelog_version_bump_displays(monkeypatch, tmp_path) -> None:
    """版本更新时显示 changelog 并更新记录；collapse 时一行提示。"""
    import pi_coding_agent.modes.interactive.app as app_mod

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## [0.2.0]\n\n- new item\n", encoding="utf-8")

    class _Messages:
        def get_messages(self):
            return []

    class _Settings:
        def __init__(self, collapse):
            self.recorded = None
            self._collapse = collapse

        def get_last_changelog_version(self):
            return "0.1.0"

        def set_last_changelog_version(self, version):
            self.recorded = version

        def get_collapse_changelog(self):
            return self._collapse

    class _Chat:
        def __init__(self):
            self.messages = []

        def add_message_agent(self, message):
            self.messages.append(message)

    class _App:
        def __init__(self, collapse):
            self._session = _Messages()
            self._settings_manager = _Settings(collapse)
            self._chat = _Chat()

    import pi_coding_agent._config as config_mod

    monkeypatch.setattr(config_mod, "get_changelog_path", lambda: changelog)

    full = _App(collapse=False)
    app_mod.PiTuiApp._show_startup_changelog_if_needed(full)
    assert full._settings_manager.recorded == "0.2.0"
    assert full._chat.messages
    assert full._chat.messages[0]["role"] == "changelog"
    assert "- new item" in full._chat.messages[0]["content"]

    collapsed = _App(collapse=True)
    app_mod.PiTuiApp._show_startup_changelog_if_needed(collapsed)
    assert "Updated to v0.2.0" in collapsed._chat.messages[0]["content"]
    assert "- new item" not in collapsed._chat.messages[0]["content"]


def test_startup_changelog_skips_resumed_session(monkeypatch, tmp_path) -> None:
    """已有消息的恢复会话不显示 changelog。"""
    import pi_coding_agent.modes.interactive.app as app_mod

    class _Messages:
        def get_messages(self):
            return [{"role": "user", "content": "hi"}]

    class _App:
        _session = _Messages()
        _settings_manager = None
        _chat = None

    called = []
    import pi_coding_agent._config as config_mod

    monkeypatch.setattr(
        config_mod, "get_changelog_path", lambda: (_ for _ in ()).throw(AssertionError())
    )
    app = _App()
    app_mod.PiTuiApp._show_startup_changelog_if_needed(app)
    assert called == []
