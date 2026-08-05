"""系统提示构建器与上下文文件加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_coding_agent.system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
    find_git_paths,
    find_shadowed_context_file,
    load_project_context_files,
    tool_snippets_for,
)


class _FakeTool:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


def _snippets():
    return {
        "read": "Read a file",
        "bash": "Run a shell command",
        "edit": "Edit a file",
        "write": "Write a file",
    }


class TestBuildSystemPrompt:
    def test_default_prompt_contains_cwd_and_tools(self, tmp_path):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                tool_snippets=_snippets(),
            )
        )
        assert "Current working directory:" in prompt
        assert str(tmp_path).replace("\\", "/") in prompt
        assert "- read: Read a file" in prompt
        assert "Be concise in your responses" in prompt

    def test_default_prompt_points_to_package_docs(self, tmp_path, monkeypatch):
        (tmp_path / "README.md").write_text("# pi", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "examples").mkdir()
        monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path))
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path / "work"),
                tool_snippets=_snippets(),
            )
        )
        assert "Pi documentation" in prompt
        assert f"Main documentation: {tmp_path / 'README.md'}" in prompt
        assert f"Additional docs: {tmp_path / 'docs'}" in prompt
        assert f"Examples: {tmp_path / 'examples'} (extensions, custom tools, SDK)" in prompt
        assert "extensions (docs/extensions.md, examples/extensions/)" in prompt
        assert "Always read pi .md files completely" in prompt

    def test_docs_section_ignores_cwd_readme(self, tmp_path, monkeypatch):
        pkg_dir = tmp_path / "pkg"
        cwd = tmp_path / "project"
        pkg_dir.mkdir()
        cwd.mkdir()
        (pkg_dir / "README.md").write_text("# pi", encoding="utf-8")
        (pkg_dir / "docs").mkdir()
        (pkg_dir / "examples").mkdir()
        (cwd / "README.md").write_text("# unrelated project", encoding="utf-8")
        monkeypatch.setenv("PI_PACKAGE_DIR", str(pkg_dir))
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(cwd),
                tool_snippets=_snippets(),
            )
        )
        assert f"Main documentation: {pkg_dir / 'README.md'}" in prompt
        assert "# unrelated project" not in prompt

    def test_custom_prompt_omits_docs_section(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path))
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                custom_prompt="You are custom.",
            )
        )
        assert "Pi documentation" not in prompt

    def test_no_tools_lists_none(self, tmp_path):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                selected_tools=[],
                tool_snippets=_snippets(),
            )
        )
        assert "Available tools:\n(none)" in prompt

    def test_custom_prompt_keeps_context_and_skills(self, tmp_path):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                custom_prompt="You are custom.",
                context_files=[
                    {"path": str(tmp_path / "AGENTS.md"), "content": "Follow repo rules"}
                ],
                skills=[],
            )
        )
        assert "You are custom." in prompt
        assert "<project_context>" in prompt
        assert "Follow repo rules" in prompt
        assert "Current working directory:" in prompt

    def test_append_system_prompt(self, tmp_path):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                append_system_prompt="Extra instruction.",
            )
        )
        assert "Extra instruction." in prompt

    def test_skills_included_when_read_available(self, tmp_path):
        from pi_coding_agent.skills import Skill

        skill = Skill(
            name="alpha",
            description="Alpha skill",
            file_path=str(tmp_path / "SKILL.md"),
            base_dir=str(tmp_path),
            source="user",
        )
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                tool_snippets=_snippets(),
                skills=[skill],
            )
        )
        assert "<available_skills>" in prompt
        assert "<name>alpha</name>" in prompt

    def test_skills_omitted_without_read(self, tmp_path):
        from pi_coding_agent.skills import Skill

        skill = Skill(
            name="alpha",
            description="Alpha skill",
            file_path=str(tmp_path / "SKILL.md"),
            base_dir=str(tmp_path),
            source="user",
        )
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                selected_tools=["bash"],
                tool_snippets={"bash": "Run a shell command"},
                skills=[skill],
            )
        )
        assert "<available_skills>" not in prompt


class TestContextFiles:
    def test_loads_agent_dir_and_ancestors(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj" / "sub"
        agent_dir.mkdir(parents=True)
        project.mkdir(parents=True)
        (agent_dir / "AGENTS.md").write_text("global rules", encoding="utf-8")
        (tmp_path / "CLAUDE.md").write_text("repo rules", encoding="utf-8")

        files = load_project_context_files(project, agent_dir)
        contents = [entry["content"] for entry in files]
        assert "global rules" in contents
        assert "repo rules" in contents
        # 全局在前、祖先随后（对齐 TS loadProjectContextFiles）。
        assert contents[0] == "global rules"
        assert contents[-1] == "repo rules"

    def test_dedupes_same_file_via_ancestors(self, tmp_path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text("rules", encoding="utf-8")
        files = load_project_context_files(agent_dir, agent_dir)
        assert len(files) == 1


class TestResourceLoaders:
    def _skill(self, path, name: str) -> None:
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} skill\n---\nBody",
            encoding="utf-8",
        )

    def test_skill_loader_only_explicit(self, tmp_path):
        from pi_coding_agent.skills import SkillLoader

        global_dir = tmp_path / "global"
        explicit = tmp_path / "explicit"
        self._skill(global_dir / "g", "global-skill")
        self._skill(explicit, "explicit-skill")

        loader = SkillLoader(global_dir=global_dir)
        only = loader.load(explicit_paths=[str(explicit)], only_explicit=True)
        assert [s.name for s in only.skills] == ["explicit-skill"]

        both = loader.load(explicit_paths=[str(explicit)], only_explicit=False)
        names = {s.name for s in both.skills}
        assert names == {"global-skill", "explicit-skill"}

    def test_template_loader_only_explicit(self, tmp_path):
        from pi_coding_agent.prompt_templates import PromptTemplateLoader

        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / "review.md").write_text(
            "---\ndescription: Review code\n---\nReview the diff.",
            encoding="utf-8",
        )
        explicit = tmp_path / "explicit.md"
        explicit.write_text("---\ndescription: Custom\n---\nCustom template.", encoding="utf-8")

        loader = PromptTemplateLoader(global_dir=global_dir)
        only = loader.load(explicit_paths=[str(explicit)], only_explicit=True)
        assert [t.name for t in only] == ["explicit"]

        both = loader.load(explicit_paths=[str(explicit)], only_explicit=False)
        names = {t.name for t in both}
        assert names == {"review", "explicit"}


class TestConfigPaths:
    def test_agent_dir_env_overrides(self, tmp_path, monkeypatch):
        from pi_coding_agent._config import get_agent_dir, get_sessions_dir

        agent_dir = tmp_path / "agent"
        session_dir = tmp_path / "sessions"
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
        monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", str(session_dir))
        assert get_agent_dir() == agent_dir.resolve()
        assert get_sessions_dir() == session_dir.resolve()

    def test_cli_sets_pi_coding_agent_marker(self, monkeypatch):
        import os

        import pytest
        from pi_coding_agent._cli import main

        monkeypatch.delenv("PI_CODING_AGENT", raising=False)
        with pytest.raises(SystemExit):
            main(["--version"])
        assert os.environ.get("PI_CODING_AGENT") == "true"

    def test_resolve_preset(self):
        from pi_coding_agent._cli import _create_parser, _resolve_preset

        parser = _create_parser()
        settings = {"presets": {"work": {"model": "m1", "tools": ["read"]}}}
        assert _resolve_preset(parser.parse_args(["--preset", "work"]), settings) == {
            "model": "m1",
            "tools": ["read"],
        }
        assert _resolve_preset(parser.parse_args([]), settings) is None
        with pytest.raises(ValueError):
            _resolve_preset(parser.parse_args(["--preset", "nope"]), settings)

    def test_allow_model_network_offline_env(self, monkeypatch):
        from pi_coding_agent._cli import _allow_model_network

        assert _allow_model_network() is True
        for value in ("1", "true", "yes", "TRUE"):
            monkeypatch.setenv("PI_OFFLINE", value)
            assert _allow_model_network() is False
        monkeypatch.setenv("PI_OFFLINE", "0")
        assert _allow_model_network() is True

    def test_no_context_files(self, tmp_path):
        files = load_project_context_files(tmp_path / "empty" / "x", tmp_path / "agent")
        assert files == []


class TestToolSnippets:
    def test_first_line_used(self):
        tools = [
            _FakeTool("read", "Read a file\nwith more details"),
            _FakeTool("bash", ""),
        ]
        snippets = tool_snippets_for(tools)
        assert snippets == {"read": "Read a file"}


class TestSessionRebuild:
    async def test_rebuild_updates_agent_state(self, tmp_path):
        from pi_agent import Agent, AgentOptions
        from pi_ai import Models
        from pi_ai.providers.faux import faux_provider

        from pi_coding_agent._session import AgentSession
        from pi_coding_agent._session_manager import SessionManager
        from pi_coding_agent.auth_storage import AuthStorage
        from pi_coding_agent.model_runtime import ModelRuntime

        store = AuthStorage.in_memory()
        models = Models(credentials=store)
        models.add_provider(faux_provider().provider)
        runtime = ModelRuntime(models, store)
        model = runtime.get_model("faux", "faux-1")
        assert model is not None

        def builder() -> str:
            return build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=str(tmp_path),
                    custom_prompt="v2 prompt",
                )
            )

        agent = Agent(AgentOptions(system_prompt="v1 prompt", model=model))
        session = AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=model,
            model_runtime=runtime,
            system_prompt_builder=builder,
        )
        assert agent.state.system_prompt == "v1 prompt"
        rebuilt = session.rebuild_system_prompt()
        assert rebuilt is not None
        assert "v2 prompt" in rebuilt
        assert "Current working directory:" in rebuilt
        assert agent.state.system_prompt == rebuilt

    async def test_rebuild_without_builder_returns_none(self, tmp_path):
        from pi_agent import Agent, AgentOptions
        from pi_ai import Model

        from pi_coding_agent._session import AgentSession
        from pi_coding_agent._session_manager import SessionManager

        agent = Agent(AgentOptions(system_prompt="static", model=None))
        session = AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=Model(id="x", provider="faux", api="openai-completions"),
        )
        assert session.rebuild_system_prompt() is None
        assert agent.state.system_prompt == "static"


def _make_nested_worktree(tmp_path):
    """构造 主仓库 + 嵌套 linked worktree（sub 在主仓库目录内）。"""
    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    (main / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (main / "AGENTS.md").write_text("main rules", encoding="utf-8")

    sub = main / "sub"
    sub.mkdir()
    git_dir = main / ".git" / "worktrees" / "sub"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/sub\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..", encoding="utf-8")
    (sub / ".git").write_text("gitdir: ../.git/worktrees/sub\n", encoding="utf-8")
    (sub / "AGENTS.md").write_text("main rules", encoding="utf-8")
    return main, sub


def test_find_git_paths_regular_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("repo rules", encoding="utf-8")

    paths = find_git_paths(repo)
    assert paths is not None
    assert paths.repo_dir == repo.resolve()
    assert paths.common_git_dir == (repo / ".git").resolve()
    assert find_shadowed_context_file(repo) is None
    files = load_project_context_files(repo, agent_dir=tmp_path / "agent")
    assert [Path(f["path"]).resolve() for f in files] == [(repo / "AGENTS.md").resolve()]


def test_nested_worktree_shadows_main_repo_agents(tmp_path):
    """嵌套 linked worktree 的 AGENTS.md 与主仓库同源，只加载一份。"""
    main, sub = _make_nested_worktree(tmp_path)

    paths = find_git_paths(sub)
    assert paths is not None
    assert paths.repo_dir == sub.resolve()
    assert paths.common_git_dir == (main / ".git").resolve()
    assert find_shadowed_context_file(sub) == str((main / "AGENTS.md").resolve())

    files = load_project_context_files(sub, agent_dir=tmp_path / "agent")
    # worktree 自己的副本保留，主仓库祖先副本被遮蔽，总共只加载一份。
    assert [Path(f["path"]).resolve() for f in files] == [(sub / "AGENTS.md").resolve()]
    assert len(files) == 1


def test_sibling_worktree_not_shadowed(tmp_path):
    """主仓库外的兄弟 worktree：不遮蔽，各自加载自己的 AGENTS.md。"""
    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    (main / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (main / "AGENTS.md").write_text("main rules", encoding="utf-8")

    feat = tmp_path / "feat"
    feat.mkdir()
    git_dir = main / ".git" / "worktrees" / "feat"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/feat\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..", encoding="utf-8")
    (feat / ".git").write_text("gitdir: ../main/.git/worktrees/feat\n", encoding="utf-8")
    (feat / "AGENTS.md").write_text("feat rules", encoding="utf-8")

    assert find_shadowed_context_file(feat) is None
    files = load_project_context_files(feat, agent_dir=tmp_path / "agent")
    assert [Path(f["path"]).resolve() for f in files] == [(feat / "AGENTS.md").resolve()]
