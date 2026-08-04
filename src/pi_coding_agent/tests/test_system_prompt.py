"""系统提示构建器与上下文文件加载测试。"""

from __future__ import annotations

import pytest

from pi_coding_agent.system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
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
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            tool_snippets=_snippets(),
        ))
        assert "Current working directory:" in prompt
        assert str(tmp_path).replace("\\", "/") in prompt
        assert "- read: Read a file" in prompt
        assert "Be concise in your responses" in prompt

    def test_no_tools_lists_none(self, tmp_path):
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            selected_tools=[],
            tool_snippets=_snippets(),
        ))
        assert "Available tools:\n(none)" in prompt

    def test_custom_prompt_keeps_context_and_skills(self, tmp_path):
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            custom_prompt="You are custom.",
            context_files=[
                {"path": str(tmp_path / "AGENTS.md"), "content": "Follow repo rules"}
            ],
            skills=[],
        ))
        assert "You are custom." in prompt
        assert "<project_context>" in prompt
        assert "Follow repo rules" in prompt
        assert "Current working directory:" in prompt

    def test_append_system_prompt(self, tmp_path):
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            append_system_prompt="Extra instruction.",
        ))
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
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            tool_snippets=_snippets(),
            skills=[skill],
        ))
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
        prompt = build_system_prompt(BuildSystemPromptOptions(
            cwd=str(tmp_path),
            selected_tools=["bash"],
            tool_snippets={"bash": "Run a shell command"},
            skills=[skill],
        ))
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
        from pi_ai import Model, Models
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
            return build_system_prompt(BuildSystemPromptOptions(
                cwd=str(tmp_path),
                custom_prompt="v2 prompt",
            ))

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
