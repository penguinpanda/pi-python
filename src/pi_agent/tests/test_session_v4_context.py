"""v4 上下文构建与 Session 兼容方法测试。"""

from __future__ import annotations

import pytest

from pi_agent.session.v4.context import SessionContextBuildOptions
from pi_agent.session.v4.memory import InMemorySessionRepo

from test_session_v4_conformance import (
    create_assistant_message,
    create_user_message,
    entry_ids,
)


@pytest.fixture
def repo() -> InMemorySessionRepo:
    return InMemorySessionRepo()


class TestBuildContext:
    @pytest.mark.asyncio
    async def test_compaction_projection_and_state(self, repo):
        session = await repo.create({"id": "s"})
        await session.append_message(create_user_message("root"))
        await session.append_message(create_assistant_message("shared"))
        await session.append_thinking_level_change("high")
        await session.append_model_change("deepseek", "deepseek-v4-pro")
        await session.append_active_tools_change(["read", "bash"])
        await session.append_compaction("checkpoint", tokens_before=100)
        await session.append_message(create_user_message("tail"))

        context = await session.build_context()
        assert context["thinkingLevel"] == "high"
        assert context["model"] == {"provider": "deepseek", "modelId": "deepseek-v4-pro"}
        assert context["activeToolNames"] == ["read", "bash"]
        assert [message["role"] for message in context["messages"]] == [
            "compactionSummary",
            "user",
        ]
        assert context["messages"][0]["summary"] == "checkpoint"
        assert context["messages"][1]["content"] == [{"type": "text", "text": "tail"}]

    @pytest.mark.asyncio
    async def test_compaction_derives_retained_tail_from_first_kept(self, repo):
        session = await repo.create({"id": "s"})
        root = await session.append_message(create_user_message("root"))
        await session.append_compaction("checkpoint", first_kept_entry_id=root)

        context = await session.build_context()
        assert [message["role"] for message in context["messages"]] == [
            "compactionSummary",
            "user",
        ]
        assert context["messages"][1]["content"] == [{"type": "text", "text": "root"}]

    @pytest.mark.asyncio
    async def test_branch_summary_after_move_to(self, repo):
        session = await repo.create({"id": "s"})
        root = await session.append_message(create_user_message("root"))
        tail = await session.append_message(create_user_message("tail"))

        summary_id = await session.move_to(root, {"summary": "branch done"})
        assert summary_id is not None
        assert await session.get_leaf_id() == summary_id
        context = await session.build_context()
        assert [message["role"] for message in context["messages"]] == [
            "user",
            "branchSummary",
        ]
        assert context["messages"][1]["summary"] == "branch done"
        assert context["messages"][0]["content"] == [{"type": "text", "text": "root"}]
        assert await session.get_entry(tail) is not None

    @pytest.mark.asyncio
    async def test_custom_projector(self, repo):
        session = await repo.create({"id": "s"})
        await session.append_custom_entry("note", {"value": 1})

        without = await session.build_context()
        assert without["messages"] == []

        def _projector(entry, index, entries):
            return [create_user_message("note")]

        options = SessionContextBuildOptions(entry_projectors={"note": _projector})
        with_projector = await session.build_context(options)
        assert [message["content"] for message in with_projector["messages"]] == [
            [{"type": "text", "text": "note"}]
        ]

    @pytest.mark.asyncio
    async def test_deferred_assistant_message_skipped(self, repo):
        session = await repo.create({"id": "s"})
        deferred = create_assistant_message("deferred")
        deferred["stopReason"] = "deferred"
        await session.append_entry(
            {"type": "message", "id": "deferred", "message": deferred}, "main"
        )

        context = await session.build_context()
        assert context["messages"] == []

    @pytest.mark.asyncio
    async def test_custom_message_passthrough(self, repo):
        session = await repo.create({"id": "s"})
        await session.append_custom_message_entry("note", "hello", display=True)

        context = await session.build_context()
        assert len(context["messages"]) == 1
        message = context["messages"][0]
        assert message["role"] == "custom"
        assert message["customType"] == "note"
        assert message["content"] == "hello"


class TestCompatMethods:
    @pytest.mark.asyncio
    async def test_branch_labels_name_stats(self, repo):
        session = await repo.create({"id": "s"})
        root = await session.append_message(create_user_message("root"))

        assert entry_ids(await session.get_branch()) == [root]
        await session.append_label(root, "keep")
        assert await session.get_label(root) == "keep"
        await session.append_session_name("My Session")
        assert await session.get_session_name() == "My Session"
        assert (await session.get_session_stats())["messageCount"] == 1

    @pytest.mark.asyncio
    async def test_build_context_entries_exposes_transform(self, repo):
        session = await repo.create({"id": "s"})
        root = await session.append_message(create_user_message("root"))
        await session.append_compaction("checkpoint")
        await session.append_message(create_user_message("tail"))

        entries = await session.build_context_entries()
        assert [entry["id"] for entry in entries] == [
            (await session.find_entries({"type": "compaction"}))[0]["id"],
            (await session.find_entries({"type": "message", "order": "oldestFirst"}))[1]["id"],
        ]
        assert root is not None
