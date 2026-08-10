"""AgentSession × V4SessionManager：prompt/compact/navigate 写入 operation records。"""

from __future__ import annotations

import pytest

from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_provider
from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager_v4 import V4SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.compaction import CompactionSettings
from pi_coding_agent.model_runtime import ModelRuntime


def _make_runtime(model_count: int = 3) -> ModelRuntime:
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


def _plain_user_message(text: str) -> dict:
    return {"role": "user", "content": text, "timestamp": 1}


async def _make_v4_session(tmp_path, compaction_settings=None):
    runtime = _make_runtime()
    manager = await V4SessionManager.in_memory(str(tmp_path))
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        )
    )
    session = AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
        compaction_settings=compaction_settings,
    )
    return session, manager


class TestAgentSessionOperationRecords:
    @pytest.mark.asyncio
    async def test_prompt_writes_run_record(self, tmp_path):
        session, manager = await _make_v4_session(tmp_path)

        await session.prompt("hi")

        started = await manager.find_records({"type": "operation_started", "operationKind": "run"})
        finished = await manager.find_records({"type": "operation_finished"})
        assert len(started) == 1
        assert len(finished) == 1
        assert finished[0]["outcome"] == "completed"
        assert await manager.recovery_state() == "idle"

    @pytest.mark.asyncio
    async def test_compact_writes_compaction_record(self, tmp_path):
        session, manager = await _make_v4_session(
            tmp_path,
            compaction_settings=CompactionSettings(
                enabled=True, reserve_tokens=100, keep_recent_tokens=1
            ),
        )
        await session.prompt("q1")
        await session.prompt("q2")

        await session.compact()

        started = await manager.find_records(
            {"type": "operation_started", "operationKind": "compaction"}
        )
        finished = await manager.find_records({"type": "operation_finished"})
        assert len(started) == 1
        assert any(record["runId"] == started[0]["id"] for record in finished)
        assert await manager.recovery_state() == "idle"

    @pytest.mark.asyncio
    async def test_navigate_writes_navigation_record(self, tmp_path):
        session, manager = await _make_v4_session(tmp_path)
        await session.prompt("q1")
        first_entry = manager.get_entries()[0]["id"]
        await session.prompt("q2")

        moved = await session.navigate_to(first_entry, summarize=False)

        assert moved is True
        started = await manager.find_records(
            {"type": "operation_started", "operationKind": "navigation"}
        )
        assert len(started) == 1
        assert started[0]["intent"]["targetId"] == first_entry
        assert await manager.recovery_state() == "idle"

    @pytest.mark.asyncio
    async def test_suspended_run_resume_replays_prompt(self, tmp_path):
        session, manager = await _make_v4_session(tmp_path)
        run_id = await manager.start_operation(
            "run", original_prompt=[_plain_user_message("replay me")]
        )

        assert await session.recovery_state() == "suspended"
        resumed = await session.resume_suspended_operation()

        assert resumed is True
        assert await session.recovery_state() == "idle"
        started = await manager.find_records({"type": "operation_started", "operationKind": "run"})
        assert len(started) == 2
        finished = await manager.find_records({"type": "operation_finished"})
        assert any(record["runId"] == run_id for record in finished)

    @pytest.mark.asyncio
    async def test_non_run_suspended_not_resumed(self, tmp_path):
        session, manager = await _make_v4_session(tmp_path)
        await manager.start_operation("compaction", result_entry_id="c-1")

        assert await session.recovery_state() == "suspended"
        assert await session.resume_suspended_operation() is False
