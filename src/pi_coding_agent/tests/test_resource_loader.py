"""统一 resource-loader 测试。"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.resource_loader import DefaultResourceLoader
from pi_coding_agent.settings_manager import SettingsManager
from pi_tui.theme import BUILTIN_THEMES


def _seed_project(tmp_path):
    """构造全局 + 项目资源目录。返回 (agent_dir, cwd)。"""
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "proj"
    global_skill = agent_dir / "skills" / "global-skill" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    global_skill.write_text(
        "---\nname: global-skill\ndescription: Global skill\n---\nBody",
        encoding="utf-8",
    )
    review_prompt = agent_dir / "prompts" / "review.md"
    review_prompt.parent.mkdir(parents=True)
    review_prompt.write_text("---\ndescription: Review\n---\nReview {{0}}", encoding="utf-8")
    global_ext = agent_dir / "extensions" / "global.py"
    global_ext.parent.mkdir(parents=True)
    global_ext.write_text(
        "def create_extension(api):\n"
        '    api.register_command("global-cmd", {"handler": lambda ctx, args: "hi", "description": "g"})\n',
        encoding="utf-8",
    )
    (agent_dir / "AGENTS.md").write_text("global rules", encoding="utf-8")

    project_skill = cwd / ".pi" / "skills" / "project-skill" / "SKILL.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text(
        "---\nname: project-skill\ndescription: Project skill\n---\nBody",
        encoding="utf-8",
    )
    project_prompt = cwd / ".pi" / "prompts" / "project.md"
    project_prompt.parent.mkdir(parents=True)
    project_prompt.write_text("---\ndescription: Project\n---\nProject {{0}}", encoding="utf-8")
    project_ext = cwd / ".pi" / "extensions" / "project.py"
    project_ext.parent.mkdir(parents=True)
    project_ext.write_text(
        "def create_extension(api):\n"
        '    api.register_command("project-cmd", {"handler": lambda ctx, args: "hi", "description": "p"})\n',
        encoding="utf-8",
    )
    custom_theme = cwd / ".pi" / "themes" / "custom.json"
    custom_theme.parent.mkdir(parents=True)
    theme_colors = dict(BUILTIN_THEMES["dark"])
    theme_colors["accent"] = "#111111"
    custom_theme.write_text(json.dumps(theme_colors), encoding="utf-8")
    (cwd / "AGENTS.md").write_text("project rules", encoding="utf-8")
    return agent_dir, cwd


@pytest.mark.asyncio
async def test_loads_all_resources_when_trusted(tmp_path):
    agent_dir, cwd = _seed_project(tmp_path)
    loader = DefaultResourceLoader(cwd, agent_dir, project_trusted=True)
    result = await loader.load()

    names = {skill.name for skill in result.skills}
    assert names == {"global-skill", "project-skill"}
    assert {prompt.name for prompt in result.prompts} == {"review", "project"}
    assert len(result.extensions) == 2
    assert len(result.themes) >= 3  # dark + light + custom
    assert any(theme.name == "custom" for theme in result.themes)
    contents = [entry["content"] for entry in result.context_files]
    assert "global rules" in contents
    assert "project rules" in contents
    assert result.system_prompt is not None
    assert "project rules" in result.system_prompt
    assert "<name>project-skill</name>" in result.system_prompt


@pytest.mark.asyncio
async def test_untrusted_ignores_project_resources(tmp_path):
    agent_dir, cwd = _seed_project(tmp_path)
    loader = DefaultResourceLoader(cwd, agent_dir, project_trusted=False)
    result = await loader.load()

    names = {skill.name for skill in result.skills}
    assert names == {"global-skill"}
    assert {prompt.name for prompt in result.prompts} == {"review"}
    assert len(result.extensions) == 1
    assert not any(theme.name == "custom" for theme in result.themes)


@pytest.mark.asyncio
async def test_reload_toggles_project_trust(tmp_path):
    agent_dir, cwd = _seed_project(tmp_path)
    loader = DefaultResourceLoader(cwd, agent_dir, project_trusted=False)
    result = await loader.load()
    assert {skill.name for skill in result.skills} == {"global-skill"}

    result = await loader.reload(project_trusted=True)
    names = {skill.name for skill in result.skills}
    assert "project-skill" in names
    assert any(theme.name == "custom" for theme in result.themes)


@pytest.mark.asyncio
async def test_diagnostics_aggregated(tmp_path):
    agent_dir, cwd = _seed_project(tmp_path)
    (cwd / ".pi" / "extensions" / "bad.py").write_text(
        "def create_extension(api\n", encoding="utf-8"
    )
    (cwd / ".pi" / "themes" / "broken.json").write_text("{not json", encoding="utf-8")
    loader = DefaultResourceLoader(cwd, agent_dir, project_trusted=True)
    result = await loader.load()

    types = {diagnostic.type for diagnostic in result.diagnostics}
    assert "error" in types
    assert "warning" in types
    assert any(diagnostic.code == "extension_load_failed" for diagnostic in result.diagnostics)
    assert any(diagnostic.code == "theme_load_failed" for diagnostic in result.diagnostics)


@pytest.mark.asyncio
async def test_settings_manager_system_prompt_used(tmp_path):
    agent_dir, cwd = _seed_project(tmp_path)
    settings_manager = SettingsManager.in_memory(
        {"systemPrompt": "custom system", "appendSystemPrompt": ["extra rules"]},
        project_trusted=True,
    )
    loader = DefaultResourceLoader(
        cwd, agent_dir, project_trusted=True, settings_manager=settings_manager
    )
    result = await loader.load()
    assert "custom system" in result.system_prompt
    assert "extra rules" in result.system_prompt


@pytest.mark.asyncio
async def test_no_context_files_disables_agents(tmp_path):
    """no_context_files=True：AGENTS.md/CLAUDE.md 不加载，其余资源不受影响。"""
    agent_dir, cwd = _seed_project(tmp_path)
    loader = DefaultResourceLoader(cwd, agent_dir, project_trusted=True, no_context_files=True)
    result = await loader.load()

    assert result.context_files == []
    assert {skill.name for skill in result.skills} == {"global-skill", "project-skill"}
    assert {prompt.name for prompt in result.prompts} == {"review", "project"}
    assert "project rules" not in result.system_prompt
