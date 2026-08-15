"""pi_agent.compaction 缺失分支测试。"""

from __future__ import annotations

import pytest
from pi_ai.types import Model

from pi_agent.compaction import (
    CompactionError,
    CompactionPreparation,
    DEFAULT_COMPACTION_SETTINGS,
    _combine_usage,
    _content_text,
    _find_valid_cut_points,
    _generate_turn_prefix_summary,
    compact,
    find_cut_point,
    find_turn_start_index,
    generate_summary_with_usage,
    prepare_compaction,
)
from pi_agent.session.v4.memory import InMemorySessionRepo


def _model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test",
        max_tokens=4096,
        context_window=128000,
    )


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


def test_find_cut_points_and_turn_start() -> None:
    entries = [
        {"type": "branch_summary", "id": "a"},
        {"type": "custom_message", "id": "b"},
        {"type": "message", "id": "c", "message": {"role": "toolResult", "content": "x"}},
        {"type": "message", "id": "d", "message": {"role": "user", "content": "q"}},
    ]
    assert _find_valid_cut_points(entries, 0, 4) == [0, 3]
    assert find_turn_start_index(entries, 3, 0) == 3
    assert (
        find_turn_start_index(
            [{"type": "message", "id": "x", "message": {"role": "toolResult"}}],
            0,
            0,
        )
        == -1
    )


def test_find_cut_point_no_cut_points() -> None:
    entries = [{"type": "compaction", "id": "c", "summary": "s"}]
    result = find_cut_point(entries, 0, 1, 1000)
    assert result.first_kept_entry_index == 0
    assert result.turn_start_index == -1
    assert result.is_split_turn is False


def test_combine_usage_optional_fields() -> None:
    first = {
        "input": 1,
        "output": 2,
        "cache_read": 3,
        "cache_write": 4,
        "total_tokens": 10,
        "cost": {"input": 0.1, "output": 0.2, "cache_read": 0, "cache_write": 0, "total": 0.3},
        "cache_write_1h": 4,
        "reasoning": 5,
    }
    second = {
        "input": 10,
        "output": 20,
        "cache_read": 30,
        "cache_write": 40,
        "total_tokens": 100,
        "cost": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0, "total": 3},
    }
    result = _combine_usage(first, second)
    assert result["input"] == 11
    assert result["output"] == 22
    assert result["cache_write_1h"] == 4
    assert result["reasoning"] == 5
    assert result["cost"]["total"] == 3.3


def test_content_text() -> None:
    assert _content_text("plain") == "plain"
    assert _content_text([{"type": "text", "text": "a"}, {"type": "image"}]) == "a"
    assert _content_text(None) == ""


@pytest.mark.asyncio
async def test_generate_summary_with_usage_success_and_previous() -> None:
    stream = _stream_fn(
        [
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "summary"}],
                "usage": {"input": 1},
            }
        ]
    )
    ok, result = await generate_summary_with_usage(
        [{"role": "user", "content": "hi", "timestamp": 1}],
        stream,
        _model(),
        1000,
        custom_instructions="focus",
        previous_summary="old",
        signal=object(),
        thinking_level="high",
    )
    assert ok is True
    assert result["text"] == "summary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "code"),
    [
        ("aborted", "aborted"),
        ("error", "summarization_failed"),
    ],
)
async def test_generate_summary_with_usage_errors(stop_reason, code) -> None:
    stream = _stream_fn([{"stop_reason": stop_reason, "error_message": "boom"}])
    ok, result = await generate_summary_with_usage(
        [{"role": "user", "content": "hi", "timestamp": 1}],
        stream,
        _model(),
        1000,
    )
    assert ok is False
    assert isinstance(result, CompactionError)
    assert result.code == code


@pytest.mark.asyncio
async def test_turn_prefix_summary_paths() -> None:
    ok, result = await _generate_turn_prefix_summary(
        [{"role": "user", "content": "hi", "timestamp": 1}],
        _stream_fn(
            [
                {
                    "stop_reason": "end_turn",
                    "content": "prefix",
                    "usage": {"input": 2},
                }
            ]
        ),
        _model(),
        1000,
    )
    assert ok is True
    assert result["text"] == "prefix"

    ok, result = await _generate_turn_prefix_summary(
        [{"role": "user", "content": "hi", "timestamp": 1}],
        _stream_fn([{"stop_reason": "error", "error_message": "boom"}]),
        _model(),
        1000,
    )
    assert ok is False
    assert result.code == "summarization_failed"


def _preparation(**overrides) -> CompactionPreparation:
    values = {
        "first_kept_entry_id": "entry-1",
        "messages_to_summarize": [],
        "turn_prefix_messages": [],
        "retained_tail": [],
        "is_split_turn": False,
        "tokens_before": 10,
        "previous_summary": None,
        "file_ops": {"read": set(), "written": set(), "edited": set()},
        "settings": DEFAULT_COMPACTION_SETTINGS,
    }
    values.update(overrides)
    return CompactionPreparation(**values)


@pytest.mark.asyncio
async def test_compact_missing_first_kept_entry() -> None:
    ok, result = await compact(_preparation(first_kept_entry_id=""), _stream_fn([]), _model())
    assert ok is False
    assert result.code == "invalid_session"


@pytest.mark.asyncio
async def test_compact_split_turn() -> None:
    preparation = _preparation(
        is_split_turn=True,
        turn_prefix_messages=[{"role": "user", "content": "prefix", "timestamp": 1}],
        messages_to_summarize=[{"role": "user", "content": "history", "timestamp": 0}],
    )
    ok, result = await compact(
        preparation,
        _stream_fn(
            [
                {
                    "stop_reason": "end_turn",
                    "content": "history summary",
                    "usage": {"input": 1},
                },
                {
                    "stop_reason": "end_turn",
                    "content": "prefix summary",
                    "usage": {"input": 2},
                },
            ]
        ),
        _model(),
    )
    assert ok is True
    assert "history summary" in result.summary
    assert "Turn Context" in result.summary


@pytest.mark.asyncio
async def test_compact_summary_error_propagates() -> None:
    ok, result = await compact(
        _preparation(),
        _stream_fn([{"stop_reason": "error", "error_message": "boom"}]),
        _model(),
    )
    assert ok is False
    assert result.code == "summarization_failed"


@pytest.mark.asyncio
async def test_prepare_compaction_previous_compaction_details() -> None:
    session = await InMemorySessionRepo().create({})
    first_id = await session.append_message({"role": "user", "content": "q1", "timestamp": 1})
    await session.append_message({"role": "assistant", "content": "a1", "timestamp": 2})
    await session.append_compaction(
        "old summary",
        first_kept_entry_id=first_id,
        tokens_before=0,
        details={"readFiles": ["a.py"], "modifiedFiles": ["b.py"]},
    )
    await session.append_message({"role": "user", "content": "q2", "timestamp": 3})
    await session.append_message({"role": "assistant", "content": "a2", "timestamp": 4})

    entries = await session.get_branch()
    ok, preparation = prepare_compaction(
        entries,
        DEFAULT_COMPACTION_SETTINGS,
    )
    assert ok is True
    assert preparation is not None
    assert preparation.previous_summary == "old summary"
    assert "a.py" in preparation.file_ops["read"]
    assert "b.py" in preparation.file_ops["edited"]


def test_prepare_compaction_missing_entry_id() -> None:
    entries = [{"type": "message", "message": {"role": "user", "content": "q"}}]
    ok, result = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
    assert ok is False
    assert isinstance(result, CompactionError)
    assert result.code == "invalid_session"
