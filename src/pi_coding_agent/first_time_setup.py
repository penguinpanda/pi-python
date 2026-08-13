"""首次启动 TUI 向导（对齐 TS FirstTimeSetupComponent）。"""

from __future__ import annotations

import os
from typing import Any

from pi_tui.engine import App, FakeTerminal, Terminal
from pi_tui.theme import ThemeLoader


def are_experimental_features_enabled() -> bool:
    """PI_EXPERIMENTAL 门控（对齐 TS areExperimentalFeaturesEnabled）。"""
    return os.environ.get("PI_EXPERIMENTAL") == "1"


def should_run_first_time_setup() -> bool:
    """对齐 TS shouldRunFirstTimeSetup：experimental + 默认 agent 目录 +
    settings.json 不存在。"""
    if not are_experimental_features_enabled():
        return False
    if os.environ.get("PI_CODING_AGENT_DIR"):
        return False
    from ._config import get_settings_path

    try:
        return not get_settings_path().exists()
    except Exception:
        return False


class _FirstTimeSetupApp(App):
    """承载 FirstTimeSetupComponent 的最小独立 App。"""

    def __init__(self, settings_manager, terminal) -> None:
        super().__init__(terminal=terminal, size=terminal.size, ui_mode="fullscreen")
        self._settings_manager = settings_manager
        self._result: tuple[str, bool] | None = None

    def on_mount(self) -> None:
        from .modes.interactive.components import FirstTimeSetupComponent

        detected_theme = ThemeLoader().detect_terminal_background()
        component = FirstTimeSetupComponent(
            detected_theme,
            on_theme_preview=self._preview_theme,
            on_submit=self._submit,
            on_cancel=self._cancel,
        )
        self.screen.mount(component)
        self.focus(component)

    def _preview_theme(self, theme: str) -> None:
        try:
            self.screen.base_style = None
        except Exception:
            pass
        self.request_render()

    def _submit(self, theme: str, share_analytics: bool) -> None:
        self._settings_manager.set_theme(theme)
        self._settings_manager.set_global_setting("enableAnalytics", share_analytics)
        self._settings_manager.flush()
        self.exit()

    def _cancel(self) -> None:
        self.exit()


def _default_terminal(size=(80, 24)):
    try:
        return Terminal(size=size)
    except Exception:
        return FakeTerminal(size=size)


async def run_first_time_setup(
    settings_manager: Any | None = None,
    terminal=None,
) -> int:
    """运行首次设置 TUI，仅设置主题与 analytics。"""
    if settings_manager is None:
        from .settings_manager import SettingsManager

        settings_manager = SettingsManager.create(os.getcwd(), project_trusted=False)
    app = _FirstTimeSetupApp(settings_manager, terminal or _default_terminal())
    await app.run_async()
    return 0


def run_first_time_setup_sync(settings_manager=None, terminal=None) -> int:
    """同步包装（测试/外部调用）。"""
    import asyncio

    return asyncio.run(run_first_time_setup(settings_manager, terminal))


__all__ = [
    "are_experimental_features_enabled",
    "should_run_first_time_setup",
    "run_first_time_setup",
    "run_first_time_setup_sync",
]
