"""V4SessionManager operation records / 恢复 / /input 编辑测试（M4）。"""

from __future__ import annotations

import pytest

from pi_agent.session.v4.types import SessionError
from pi_coding_agent._session_manager_v4 import V4SessionManager


def _user_message(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _zero_usage() -> dict:
    return {
        "input": 10,
        "output": 5,
        "cache_read": 3,
        "cache_write": 2,
        "total_tokens": 20,
        "cost": {
            "input": 1,
            "output": 2,
            "cache_read": 3,
            "cache_write": 4,
            "total": 10,
        },
    }


def _plain_user_message(text: str) -> dict:
    return {"role": "user", "content": text, "timestamp": 1}


class TestOperationRecords:
    @pytest.mark.asyncio
    async def test_run_lifecycle(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="rec"
        )
        assert await manager.recovery_state() == "idle"

        run_id = await manager.start_operation("run")
        assert await manager.recovery_state() == "suspended"
        operations = await manager.open_operations()
        assert operations[0]["id"] == run_id
        assert operations[0]["intent"]["kind"] == "run"

        await manager.finish_operation(run_id)
        assert await manager.recovery_state() == "idle"
        assert await manager.open_operations() == []

    @pytest.mark.asyncio
    async def test_second_open_operation_rejected(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="rec"
        )
        await manager.start_operation("run")
        with pytest.raises(SessionError) as excinfo:
            await manager.start_operation("run")
        assert excinfo.value.code == "storage"

    @pytest.mark.asyncio
    async def test_kinds_and_filter(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="rec"
        )
        compaction_id = await manager.start_operation("compaction", result_entry_id="c-1")
        compact_records = await manager.find_records(
            {"type": "operation_started", "operationKind": "compaction"}
        )
        assert [record["id"] for record in compact_records] == [compaction_id]
        await manager.finish_operation(compaction_id)

        navigation_id = await manager.start_operation("navigation", target_id="t-1", summarize=True)
        navigation_records = await manager.find_records(
            {"type": "operation_started", "operationKind": "navigation"}
        )
        assert navigation_records[0]["id"] == navigation_id
        assert navigation_records[0]["intent"]["summarize"] is True
        await manager.finish_operation(navigation_id)

    @pytest.mark.asyncio
    async def test_usage_record_updates_stats(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="rec"
        )
        usage = _zero_usage()
        await manager.record_usage(cause="assistant", usage=usage, run_id="run")
        await manager.record_usage(
            cause="adjustment",
            usage={
                "input": -2,
                "output": 0,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": -2,
                "cost": {
                    "input": -0.5,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "total": -0.5,
                },
            },
        )

        stats = await manager.get_session_stats()
        assert stats["messageCount"] == 0
        assert stats["cachedTokens"] == 3
        assert stats["uncachedTokens"] == 10
        assert stats["totalTokens"] == 18
        assert stats["costTotal"] == 9.5

    @pytest.mark.asyncio
    async def test_reopen_detects_suspended_and_recovers(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="rec"
        )
        run_id = await manager.start_operation("run")

        reopened = await V4SessionManager.open(manager.session_path)
        assert await reopened.recovery_state() == "suspended"
        await reopened.finish_operation(run_id)

        again = await V4SessionManager.open(manager.session_path)
        assert await again.recovery_state() == "idle"


class TestEditMessage:
    @pytest.mark.asyncio
    async def test_merge_appends_and_moves_lane(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="edit"
        )
        entry_id = await manager.append_message(_plain_user_message("hello"))
        await manager.append_message(_plain_user_message("ignored"))

        merged = await manager.edit_message(entry_id, "world")

        assert merged == "hello\n\nworld"
        messages = [m for m in manager.get_entries() if m["type"] == "message"]
        assert len(messages) == 3
        assert manager.get_leaf_id() == messages[-1]["id"]
        assert manager.build_context()[-1]["content"] == "hello\n\nworld"

        reopened = await V4SessionManager.open(manager.session_path)
        assert reopened.build_context()[-1]["content"] == "hello\n\nworld"
        assert reopened.get_entry(entry_id) is not None

    @pytest.mark.asyncio
    async def test_replace_mode(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="edit"
        )
        entry_id = await manager.append_message(_plain_user_message("hello"))

        merged = await manager.edit_message(entry_id, "replaced", mode="replace")

        assert merged == "replaced"
        assert manager.build_context()[-1]["content"] == "replaced"

    @pytest.mark.asyncio
    async def test_invalid_targets_rejected(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="edit"
        )
        await manager.append_message(_user_message("hi"))
        with pytest.raises(ValueError, match="Entry not found"):
            await manager.edit_message("missing", "x")
        assistant_id = await manager.append_message(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "a"}],
                "timestamp": 2,
            }
        )
        with pytest.raises(ValueError, match="not a user message"):
            await manager.edit_message(assistant_id, "x")
