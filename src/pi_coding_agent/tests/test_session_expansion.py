"""AgentSession 管道集成测试：/skill:name 与 /templateName 展开。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.prompt_templates import PromptTemplateLoader
from pi_coding_agent.skills import SkillLoader


def _make_session(tmp_path: Path, skill_loader=None, template_loader=None) -> AgentSession:
    store_holder = {}

    from pi_coding_agent.auth_storage import AuthStorage

    models = Models(credentials=AuthStorage.in_memory())
    core = faux_provider()

    async def factory(context, _options, _state, _model):
        store_holder["messages"] = list(context.messages)
        return faux_assistant_message("ok")

    core.set_responses([factory])
    models.add_provider(core.provider)
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=models.stream,
        )
    )
    session = AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        skill_loader=skill_loader,
        template_loader=template_loader,
    )
    return session, store_holder


def _first_user_text(messages) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            parts = [
                block.get("text", "")
                for block in content or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts)
    return ""


@pytest.mark.asyncio
async def test_skill_command_expansion(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "greet").mkdir(parents=True)
    (skills_dir / "greet" / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Greet skill\n---\n\nSkill body content",
        encoding="utf-8",
    )
    loader = SkillLoader(global_dir=skills_dir)
    loader.load()
    session, holder = _make_session(tmp_path, skill_loader=loader)

    await session.prompt("/skill:greet please do it")
    await session.wait_for_idle()
    text = _first_user_text(holder["messages"])
    assert '<skill name="greet"' in text
    assert "Skill body content" in text
    assert "please do it" in text
    await session.dispose()


@pytest.mark.asyncio
async def test_template_command_expansion(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "review.md").write_text(
        "---\ndescription: Review template\n---\n\nPlease review $1",
        encoding="utf-8",
    )
    loader = PromptTemplateLoader(global_dir=prompts_dir)
    loader.load()
    session, holder = _make_session(tmp_path, template_loader=loader)

    await session.prompt("/review the-new-file.py")
    await session.wait_for_idle()
    assert _first_user_text(holder["messages"]) == "Please review the-new-file.py"
    await session.dispose()


@pytest.mark.asyncio
async def test_unknown_slash_passes_through(tmp_path):
    session, holder = _make_session(tmp_path)
    await session.prompt("/not-a-template anything")
    await session.wait_for_idle()
    assert _first_user_text(holder["messages"]) == "/not-a-template anything"
    await session.dispose()


@pytest.mark.asyncio
async def test_expand_prompt_returns_original_for_non_slash(tmp_path):
    session, _holder = _make_session(tmp_path)
    assert session.expand_prompt("plain text") == "plain text"
