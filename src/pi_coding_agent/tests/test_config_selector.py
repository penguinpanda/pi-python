"""pi config 资源选择器测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_coding_agent.config_selector import (
    build_config_selector_model,
    persist_resource_toggle,
    run_config_selector,
)
from pi_coding_agent.settings_manager import SettingsManager
from pi_tui.engine import FakeTerminal


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_build_and_persist_top_level_resource(tmp_path):
    agent_dir = tmp_path / "agent"
    skill = _write(agent_dir / "skills" / "demo" / "SKILL.md", "---\ndescription: demo\n---\n")
    manager = SettingsManager.create(tmp_path, agent_dir=agent_dir)

    model = build_config_selector_model(
        cwd=tmp_path,
        agent_dir=agent_dir,
        settings_manager=manager,
        write_scope="global",
    )
    item = next(
        item for group in model.groups for item in group.items if item.resource_type == "skills"
    )
    assert item.enabled is True

    persist_resource_toggle(manager, item, False)
    assert Path(agent_dir / "settings.json").is_file()
    assert manager.get_skills() == [f"-skills/{skill.parent.name}"]


def test_package_resource_persists_filters(tmp_path):
    agent_dir = tmp_path / "agent"
    manager = SettingsManager.create(tmp_path, agent_dir=agent_dir)
    manager.set_global_setting("packages", ["npm:demo"])
    pkg_root = agent_dir / "packages" / "demo"
    _write(pkg_root / "prompts" / "hello.md", "# Hello")

    model = build_config_selector_model(
        cwd=tmp_path,
        agent_dir=agent_dir,
        settings_manager=manager,
        write_scope="global",
    )
    item = next(item for group in model.groups for item in group.items if item.origin == "package")
    persist_resource_toggle(manager, item, False)
    packages = manager.get_global_settings()["packages"]
    assert packages == [{"source": "npm:demo", "prompts": ["-prompts/hello.md"]}]


@pytest.mark.asyncio
async def test_config_selector_tui_exits_on_escape(tmp_path):
    agent_dir = tmp_path / "agent"
    _write(agent_dir / "skills" / "demo" / "SKILL.md", "---\ndescription: demo\n---\n")
    manager = SettingsManager.create(tmp_path, agent_dir=agent_dir)
    term = FakeTerminal(size=(80, 24))

    async def _run() -> None:
        await run_config_selector(
            manager,
            cwd=tmp_path,
            agent_dir=agent_dir,
            write_scope="global",
            terminal=term,
        )

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.05)
    term.feed(b"\x1b")
    await asyncio.wait_for(task, timeout=1)
