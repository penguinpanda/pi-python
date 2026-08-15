"""pi_agent.branch_summarization 补充测试。"""

from __future__ import annotations

import pytest
from pi_ai.types import Model

from pi_agent.branch_summarization import (
    BranchSummaryError,
    _get_message_from_entry,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    prepare_branch_entries,
)
from pi_agent.session.v4.types import SessionError


class _Stream:
    def __init__(self, response) -> None:
        self._response = response

    async def result(self):
        return self._response


def _stream_fn(responses):
    iterator = iter(responses)

    async def stream_fn(_model, _context, _options):
        return _Stream(next(iterator))

    return stream_fn


def _model() -> Model:
    return Model(
        id="test",
        provider="test",
        api="openai-completions",
        name="Test",
        context_window=128000,
    )


def _entry(entry_type: str, **overrides) -> dict:
    base = {
        "type": entry_type,
        "id": overrides.pop("id", f"{entry_type}-1"),
        "parentId": overrides.pop("parentId", None),
        "timestamp": overrides.pop("timestamp", 1),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_collect_entries_no_old_leaf() -> None:
    result = await collect_entries_for_branch_summary(object(), None, "target")
    assert result == {"entries": [], "commonAncestorId": None}


@pytest.mark.asyncio
async def test_collect_entries_success_and_missing() -> None:
    class _Session:
        def __init__(self, entries) -> None:
            self._by_id = {entry["id"]: entry for entry in entries}

        async def get_branch(self, leaf_id):
            if leaf_id == "old":
                return [self._by_id["root"], self._by_id["old"]]
            return [self._by_id["root"], self._by_id["target"]]

        async def get_entry(self, entry_id):
            return self._by_id.get(entry_id)

    entries = [
        _entry("message", id="root", message={"role": "user", "content": "root"}),
        _entry("message", id="old", message={"role": "user", "content": "old"}),
        _entry("message", id="target", message={"role": "user", "content": "target"}),
    ]
    session = _Session(entries)
    result = await collect_entries_for_branch_summary(session, "old", "target")
    assert result["commonAncestorId"] == "root"
    assert [entry["id"] for entry in result["entries"]] == ["old"]

    class _Missing:
        async def get_branch(self, leaf_id):
            return [{"id": "root"}]

        async def get_entry(self, entry_id):
            return None

    with pytest.raises(SessionError, match="not found"):
        await collect_entries_for_branch_summary(_Missing(), "old", "root")


def test_get_message_from_entry_variants() -> None:
    assert (
        _get_message_from_entry(_entry("message", message={"role": "toolResult", "content": "x"}))
        is None
    )
    assert (
        _get_message_from_entry(_entry("message", message={"role": "user", "content": "q"}))
        is not None
    )
    assert (
        _get_message_from_entry(
            _entry("custom_message", customType="note", data={"x": 1}, display=True)
        )
        is None
    )
    assert _get_message_from_entry(_entry("branch_summary", summary="s", fromId="old")) is not None
    assert _get_message_from_entry(_entry("compaction", summary="s", tokensBefore=10)) is not None
    assert _get_message_from_entry(_entry("custom", customType="x")) is None


def test_prepare_branch_entries_budget_and_details() -> None:
    entries = [
        _entry("message", id="user", message={"role": "user", "content": "q" * 100}),
        _entry(
            "branch_summary",
            id="summary",
            summary="old",
            fromId="x",
            details={"readFiles": ["a.py"], "modifiedFiles": ["b.py"]},
        ),
        _entry(
            "compaction",
            id="compact",
            summary="compact",
            tokensBefore=5,
        ),
    ]
    prepared = prepare_branch_entries(entries, token_budget=10)
    assert prepared["messages"]
    assert "a.py" in prepared["fileOps"]["read"]
    assert "b.py" in prepared["fileOps"]["edited"]


@pytest.mark.asyncio
async def test_generate_branch_summary_no_messages() -> None:
    ok, result = await generate_branch_summary([], stream_fn=_stream_fn([]), model=_model())
    assert ok is True
    assert result["summary"] == "No content to summarize"


@pytest.mark.asyncio
async def test_generate_branch_summary_success_and_instructions() -> None:
    entries = [_entry("message", message={"role": "user", "content": "hello"})]
    ok, result = await generate_branch_summary(
        entries,
        stream_fn=_stream_fn(
            [
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "branch summary"}],
                    "usage": {"input": 1},
                }
            ]
        ),
        model=_model(),
        custom_instructions="focus",
        replace_instructions=True,
    )
    assert ok is True
    assert "Summary of that exploration" in result["summary"]
    assert result["usage"] == {"input": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "code"),
    [("aborted", "aborted"), ("error", "summarization_failed")],
)
async def test_generate_branch_summary_errors(stop_reason, code) -> None:
    entries = [_entry("message", message={"role": "user", "content": "hello"})]
    ok, result = await generate_branch_summary(
        entries,
        stream_fn=_stream_fn([{"stop_reason": stop_reason, "error_message": "boom"}]),
        model=_model(),
    )
    assert ok is False
    assert isinstance(result, BranchSummaryError)
    assert result.code == code
