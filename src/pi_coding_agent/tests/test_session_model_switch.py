"""AgentSession 模型切换测试（set_model / cycle_model / thinking level）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_resolver import ScopedModel
from pi_coding_agent.model_runtime import ModelRuntime


def _make_runtime_with_models(model_count: int = 3) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    models_list = [
        Model(
            id=f"faux-{index}",
            provider="faux",
            api="openai-completions",
            name=f"Faux {index}",
            reasoning=(index % 2 == 0),
        )
        for index in range(1, model_count + 1)
    ]
    core = faux_provider(models=models_list)
    models.add_provider(core.provider)
    return ModelRuntime(models, store)


def _make_session(
    runtime: ModelRuntime,
    tmp_path: Path,
    *,
    scoped_models: list[ScopedModel] | None = None,
) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            thinking_level="off",
        )
    )
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
        scoped_models=scoped_models,
    )


class TestSetModel:
    async def test_set_model_updates_agent_and_session(self, tmp_path):
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)
        events: list[dict] = []
        session.subscribe(lambda event: events.append(event))

        next_model = runtime.get_model("faux", "faux-2")
        assert next_model is not None
        await session.set_model(next_model)

        assert session.model is not None
        assert session.model.id == "faux-2"
        assert session._agent.state.model.id == "faux-2"

        entries = session._session_manager.get_entries()
        assert entries[-1]["type"] == "model_change"
        assert entries[-1]["provider"] == "faux"
        assert entries[-1]["modelId"] == "faux-2"

        assert any(event["type"] == "model_changed" for event in events)
        await session.dispose()

    async def test_set_model_validates_auth(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)
        from pi_ai.providers.openai import openai_provider

        runtime.register_native_provider(openai_provider())
        unknown = runtime.get_model("openai", "gpt-5-chat-latest")
        assert unknown is not None
        with pytest.raises(RuntimeError, match="No API key"):
            await session.set_model(unknown)
        await session.dispose()


class TestCycleModel:
    async def test_cycle_scoped_models(self, tmp_path):
        runtime = _make_runtime_with_models()
        scoped = [ScopedModel(runtime.get_model("faux", f"faux-{index}")) for index in (1, 2, 3)]
        session = _make_session(runtime, tmp_path, scoped_models=scoped)

        result = await session.cycle_model(1)
        assert result is not None
        assert result.is_scoped is True
        assert result.model.id == "faux-2"

        result = await session.cycle_model(1)
        assert result.model.id == "faux-3"

        # 环绕
        result = await session.cycle_model(1)
        assert result.model.id == "faux-1"

        # 反向
        result = await session.cycle_model(-1)
        assert result.model.id == "faux-3"
        await session.dispose()

    async def test_cycle_available_models(self, tmp_path):
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)

        result = await session.cycle_model(1)
        assert result is not None
        assert result.is_scoped is False
        assert result.model.id == "faux-2"
        await session.dispose()

    async def test_cycle_single_model_returns_none(self, tmp_path):
        runtime = _make_runtime_with_models(model_count=1)
        session = _make_session(runtime, tmp_path)
        assert await session.cycle_model(1) is None
        await session.dispose()


class TestThinkingLevel:
    def test_available_levels_for_non_reasoning_model(self, tmp_path):
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)
        assert session.get_available_thinking_levels() == ["off"]
        assert session.supports_thinking() is False

    async def test_set_thinking_level_persists_change(self, tmp_path):
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)
        session.set_thinking_level("off")
        # off 已是默认，不产生新条目
        entries = session._session_manager.get_entries()
        assert all(entry["type"] != "thinking_level_change" for entry in entries)

        session.set_thinking_level("medium")
        # 非 reasoning 模型：clamp 到 off，无变化
        entries = session._session_manager.get_entries()
        assert all(entry["type"] != "thinking_level_change" for entry in entries)
        await session.dispose()

    async def test_thinking_level_clamps_to_model_capabilities(self, tmp_path):
        runtime = _make_runtime_with_models()
        session = _make_session(runtime, tmp_path)
        reasoning_model = runtime.get_model("faux", "faux-2")
        assert reasoning_model.reasoning is True
        await session.set_model(reasoning_model)
        assert session.supports_thinking() is True
        assert "medium" in session.get_available_thinking_levels()

        session.set_thinking_level("max")
        # max 不在基础支持集 → clamp 到 high（最高基础级别）
        assert session.thinking_level == "high"
        await session.dispose()
