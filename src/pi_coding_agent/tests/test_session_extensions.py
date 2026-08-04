"""AgentSession × 扩展集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import Extension


def _make_session(
    tmp_path: Path,
    runner: ExtensionRunner | None,
    store_holder: dict,
) -> AgentSession:
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
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        extension_runner=runner,
    )


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
async def test_input_event_transforms_prompt(tmp_path):
    def transform(event, ctx):
        return {"action": "transform", "text": f"EXT:{event['text']}"}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["input"] = [transform]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await session.prompt("hello")
    await session.wait_for_idle()
    assert _first_user_text(holder["messages"]) == "EXT:hello"
    await session.dispose()


@pytest.mark.asyncio
async def test_agent_events_forwarded_to_extensions(tmp_path):
    seen: list[str] = []

    def on_message_end(event, ctx):
        seen.append(event.get("type"))
        return None

    def on_agent_settled(event, ctx):
        seen.append(event.get("type"))
        return None

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["message_end"] = [on_message_end]
    extension.handlers["agent_settled"] = [on_agent_settled]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await session.prompt("hi")
    await session.wait_for_idle()
    assert "message_end" in seen
    assert "agent_settled" in seen
    await session.dispose()


@pytest.mark.asyncio
async def test_bind_session_provider_registration(tmp_path):
    from pi_ai import Models

    models = Models(credentials=AuthStorage.in_memory())
    models.add_provider(faux_provider().provider)
    from pi_coding_agent.model_runtime import ModelRuntime

    runtime = ModelRuntime(models, AuthStorage.in_memory())
    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.providers.append(
        (
            "acme",
            {
                "api_key": "sk-acme",
                "base_url": "https://acme.api/v1",
                "models": [{"id": "acme-1", "api": "openai-completions", "reasoning": False}],
            },
        )
    )
    runner = ExtensionRunner([extension], cwd=str(tmp_path), model_runtime=runtime)

    core = faux_provider()
    core.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=runtime.get_model("faux", "faux-1"),
            stream_fn=models.stream,
        )
    )
    session = AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=runtime.get_model("faux", "faux-1"),
        model_runtime=runtime,
        extension_runner=runner,
    )
    try:
        model = runtime.get_model("acme", "acme-1")
        assert model is not None
        assert model.base_url == "https://acme.api/v1"
    finally:
        await session.dispose()
