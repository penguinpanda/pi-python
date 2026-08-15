"""/settings 信任门控与 SettingsManager 写入路径测试。"""

from __future__ import annotations

import os
import shutil
import uuid

import pytest

from pi_coding_agent.modes.interactive.slash_commands import (
    SlashCommandRegistry,
    SlashContext,
    register_builtin_commands,
)
from pi_coding_agent.settings_manager import SettingsManager


class _FakeSession:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd


@pytest.fixture
def workdir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-slashset-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_context(
    cwd: str, manager: SettingsManager | None, notifications: list[str]
) -> SlashContext:
    return SlashContext(
        session=_FakeSession(cwd),
        notify=notifications.append,
        settings_manager=manager,
    )


async def test_settings_write_rejected_when_untrusted(workdir: str) -> None:
    manager = SettingsManager.create(workdir, workdir, project_trusted=False)
    registry = SlashCommandRegistry()
    register_builtin_commands(registry)
    notifications: list[str] = []
    await registry.execute(
        "/settings defaultModel=deepseek-chat", _make_context(workdir, manager, notifications)
    )
    assert "not trusted" in notifications[-1]
    assert not os.path.exists(os.path.join(workdir, ".pi", "settings.json"))


async def test_settings_write_allowed_when_trusted(workdir: str) -> None:
    manager = SettingsManager.create(workdir, workdir, project_trusted=True)
    registry = SlashCommandRegistry()
    register_builtin_commands(registry)
    notifications: list[str] = []
    await registry.execute(
        "/settings defaultModel=deepseek-chat", _make_context(workdir, manager, notifications)
    )
    assert "Saved defaultModel" in notifications[-1]
    assert manager.get("defaultModel") == "deepseek-chat"
    assert os.path.exists(os.path.join(workdir, ".pi", "settings.json"))
