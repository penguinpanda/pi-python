"""首次启动向导测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent.first_time_setup import run_first_time_setup
from pi_coding_agent.settings_manager import SettingsManager
from pi_tui.engine import FakeTerminal


def _settings(tmp_path) -> SettingsManager:
    return SettingsManager.create(str(tmp_path), agent_dir=tmp_path)


@pytest.mark.asyncio
async def test_setup_saves_theme_and_analytics(tmp_path):
    manager = _settings(tmp_path)
    term = FakeTerminal(size=(80, 24))

    async def _run() -> None:
        await run_first_time_setup(manager, terminal=term)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.05)
    term.feed(b"\r")  # theme -> analytics
    await asyncio.sleep(0.05)
    term.feed(b"\r")  # finish
    await asyncio.wait_for(task, timeout=1)

    assert manager.get_theme() == "dark"
    assert manager.as_dict().get("enableAnalytics") is True


@pytest.mark.asyncio
async def test_setup_cancel_does_not_require_api_key(tmp_path):
    manager = _settings(tmp_path)
    term = FakeTerminal(size=(80, 24))

    async def _run() -> None:
        await run_first_time_setup(manager, terminal=term)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.05)
    term.feed(b"\x1b")
    await asyncio.wait_for(task, timeout=1)

    assert manager.get_theme() is None
