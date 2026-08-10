"""AgentHarness × v4 Session 集成测试（M3 落地的关键验证）。"""

from __future__ import annotations

import pytest

from pi_agent._harness import AgentHarness
from pi_agent._harness_types import AgentHarnessOptions, NavigateOptions
from pi_agent.session.v4.memory import InMemorySessionRepo
from pi_agent.session.v4.session import Session

from test_harness import (
    _make_faux,
    _make_model,
    _make_models,
    _text_response,
    faux_assistant_message,
)


async def _v4_session() -> tuple[Session, InMemorySessionRepo]:
    repo = InMemorySessionRepo()
    session = await repo.create({"id": "harness"})
    return session, repo


def _options(session, *, stream_fn=None, responses=None) -> AgentHarnessOptions:
    return AgentHarnessOptions(
        model=_make_model(),
        session=session,
        models=_make_models(responses=responses, stream_fn=stream_fn),
    )


class TestHarnessWithV4Session:
    @pytest.mark.asyncio
    async def test_basic_prompt_writes_v4_entries(self):
        session, _ = await _v4_session()
        harness = AgentHarness(_options(session))

        result = await harness.prompt("Hi")

        assert result["role"] == "assistant"
        context = await session.build_context()
        assert [message["role"] for message in context["messages"]] == [
            "user",
            "assistant",
        ]
        messages = await session.find_entries({"type": "message", "order": "oldestFirst"})
        assert [entry["seq"] for entry in messages] == [1, 2]

    @pytest.mark.asyncio
    async def test_compact_generates_v4_compaction_entry(self):
        core = _make_faux(
            [
                _text_response("first answer"),
                faux_assistant_message("## Goal\ncompacted"),
            ]
        )
        session, _ = await _v4_session()
        harness = AgentHarness(_options(session, stream_fn=core.stream))

        await harness.prompt("question")
        result = await harness.compact()

        assert "## Goal" in result.summary
        entries = await session.get_branch()
        assert entries[-1]["type"] == "compaction"
        assert "retainedTail" in entries[-1]
        context = await session.build_context()
        assert context["messages"][0]["role"] == "compactionSummary"

    @pytest.mark.asyncio
    async def test_navigate_tree_moves_leaf(self):
        core = _make_faux([_text_response("a1"), _text_response("a2")])
        session, _ = await _v4_session()
        harness = AgentHarness(_options(session, stream_fn=core.stream))

        await harness.prompt("q1")
        first_leaf = await harness.get_leaf_id()
        assert first_leaf is not None
        first_entry = (await session.get_branch())[0]["id"]

        result = await harness.navigate_tree(first_entry)

        assert result.cancelled is False
        assert await harness.get_leaf_id() == first_entry
        await harness.prompt("q2")
        second_leaf = await harness.get_leaf_id()
        assert second_leaf is not None and second_leaf != first_entry
        await harness.navigate_tree(first_leaf)
        assert await harness.get_leaf_id() == first_leaf

    @pytest.mark.asyncio
    async def test_navigate_tree_summarize_generates_branch_summary(self):
        core = _make_faux(
            [
                _text_response("a1"),
                _text_response("a2"),
                faux_assistant_message("## Goal\nbranch summary"),
            ]
        )
        session, _ = await _v4_session()
        harness = AgentHarness(_options(session, stream_fn=core.stream))

        await harness.prompt("q1")
        first_leaf = await harness.get_leaf_id()
        assert first_leaf is not None
        first_entry = (await session.get_branch())[0]["id"]
        await harness.navigate_tree(first_entry)
        await harness.prompt("q2")

        result = await harness.navigate_tree(first_leaf, NavigateOptions(summarize=True))

        assert result.cancelled is False
        assert result.summary_entry is not None
        assert result.summary_entry["type"] == "branch_summary"
        context = await session.build_context()
        assert "branchSummary" in [message["role"] for message in context["messages"]]
