"""pi_coding_agent.compaction 单元测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pi_agent import set_default_stream_fn
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.utils.estimate import ContextUsageEstimate

from pi_coding_agent.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    compact,
    compute_file_lists,
    create_file_ops,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops_from_message,
    find_cut_point,
    format_file_operations,
    generate_summary_with_usage,
    get_last_assistant_usage,
    is_cut_point_message,
    is_turn_start_message,
    prepare_compaction,
    serialize_conversation,
    should_compact,
)


@pytest.fixture
def faux_env():
    """注册 Faux Provider 的 Models + 全局默认流函数。"""
    core = faux_provider()
    models = Models()
    models.add_provider(core.provider)
    set_default_stream_fn(models.stream)
    yield models, core
    set_default_stream_fn(None)


def _user(content: str, ts: int = 1) -> dict:
    return {"role": "user", "content": content, "timestamp": ts}


def _assistant(content: str, ts: int = 2, usage: dict | None = None, stop_reason: str = "stop") -> dict:
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "api": "openai-completions",
        "provider": "faux",
        "model": "faux-1",
        "stop_reason": stop_reason,
        "timestamp": ts,
    }
    if usage is not None:
        msg["usage"] = usage
    return msg


def _usage(total: int) -> dict:
    return {
        "input": total,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": total,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def _entry(msg: dict, entry_id: str = "e1", parent: str | None = None) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "message": msg,
    }


# ============================================================================
# estimate_tokens
# ============================================================================


class TestEstimateTokens:
    def test_user_string(self):
        assert estimate_tokens(_user("x" * 8)) == 2

    def test_user_blocks_with_image(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 4},
                {"type": "image", "url": "http://x", "data": None, "mime_type": "image/png"},
            ],
            "timestamp": 1,
        }
        # text 4 + image 4800 = 4804 → ceil(4804/4) = 1201
        assert estimate_tokens(msg) == 1201

    def test_assistant_blocks(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "abcd"},
                {"type": "thinking", "thinking": "wxyz"},
                {"type": "toolCall", "id": "c1", "name": "foo", "raw_arguments": "{}", "arguments": {"a": 1}},
            ],
            "api": "openai-completions",
            "provider": "faux",
            "model": "faux-1",
            "timestamp": 1,
        }
        # 4 + 4 + (3 + 7) = 18 → ceil(18/4) = 5
        assert estimate_tokens(msg) == 5

    def test_tool_result(self):
        msg = {
            "role": "toolResult",
            "tool_call_id": "c1",
            "tool_name": "t",
            "content": [{"type": "text", "text": "x" * 8}],
            "is_error": False,
            "timestamp": 1,
        }
        assert estimate_tokens(msg) == 2

    def test_compaction_summary(self):
        msg = {"role": "compactionSummary", "summary": "x" * 8, "timestamp": 1}
        assert estimate_tokens(msg) == 2

    def test_agent_role_string_content(self):
        msg = {"role": "observation", "content": "x" * 8, "timestamp": 1}
        assert estimate_tokens(msg) == 2

    def test_unknown_returns_zero(self):
        assert estimate_tokens({}) == 0


# ============================================================================
# estimate_context_tokens（AgentMessage 版）
# ============================================================================


class TestEstimateContextTokens:
    def test_no_usage_estimates_all(self):
        messages = [_user("x" * 4), _assistant("y" * 8)]
        estimate = estimate_context_tokens(messages)
        assert estimate == ContextUsageEstimate(tokens=3, usage_tokens=0, trailing_tokens=3, last_usage_index=None)

    def test_uses_last_assistant_usage(self):
        messages = [_user("x" * 4), _assistant("ok", usage=_usage(100)), _user("tail")]
        estimate = estimate_context_tokens(messages)
        # usage 100 + tail "tail"(4 字符→1 token)
        assert estimate == ContextUsageEstimate(tokens=101, usage_tokens=100, trailing_tokens=1, last_usage_index=1)

    def test_skips_aborted_error_and_zero_usage(self):
        aborted = _assistant("a", stop_reason="aborted", usage=_usage(50))
        error = _assistant("e", stop_reason="error", usage=_usage(50))
        zero = _assistant("z", usage=_usage(0))
        messages = [aborted, error, zero, _user("tail")]
        estimate = estimate_context_tokens(messages)
        assert estimate.last_usage_index is None

    def test_get_last_assistant_usage(self):
        messages = [_assistant("a", usage=_usage(10)), _user("x"), _assistant("b", usage=_usage(20))]
        usage = get_last_assistant_usage(messages)
        assert usage is not None
        assert usage["total_tokens"] == 20


# ============================================================================
# should_compact
# ============================================================================


class TestShouldCompact:
    def test_over_threshold(self):
        # 128000 - 16384 = 111616；120000 超出
        assert should_compact(120_000, 128_000, DEFAULT_COMPACTION_SETTINGS) is True

    def test_under_threshold(self):
        assert should_compact(10_000, 128_000, DEFAULT_COMPACTION_SETTINGS) is False

    def test_disabled(self):
        settings = CompactionSettings(enabled=False)
        assert should_compact(200_000, 128_000, settings) is False


# ============================================================================
# 切割点
# ============================================================================


class TestCutPoint:
    def test_is_cut_point_message(self):
        assert is_cut_point_message({"role": "user"}) is True
        assert is_cut_point_message({"role": "assistant"}) is True
        assert is_cut_point_message({"role": "toolResult"}) is False
        assert is_cut_point_message({"role": "compactionSummary"}) is True
        assert is_cut_point_message({"role": "system"}) is False

    def test_is_turn_start_message(self):
        assert is_turn_start_message({"role": "user"}) is True
        assert is_turn_start_message({"role": "assistant"}) is False
        assert is_turn_start_message({"role": "toolResult"}) is False

    def test_find_cut_point_keeps_recent(self):
        # 5 条 user 消息，每条估算 1 token；keep_recent_tokens=2 → 保留最后 2 条
        entries = [_entry(_user("a"), "e0"), _entry(_user("b"), "e1"), _entry(_user("c"), "e2"),
                   _entry(_user("d"), "e3"), _entry(_user("e"), "e4")]
        cut = find_cut_point(entries, 0, len(entries), 2)
        assert cut.first_kept_entry_index == 3
        assert cut.is_split_turn is False

    def test_find_cut_point_never_cuts_at_tool_result(self):
        entries = [
            _entry(_user("a"), "e0"),
            _entry(_assistant("t1"), "e1"),
            _entry({"role": "toolResult", "tool_call_id": "c1", "tool_name": "t",
                    "content": [{"type": "text", "text": "r"}], "is_error": False, "timestamp": 3}, "e2"),
            _entry(_user("b"), "e3"),
        ]
        # 预算很小：期望在 e3（user）切割，绝不落在 toolResult
        cut = find_cut_point(entries, 0, len(entries), 1)
        assert entries[cut.first_kept_entry_index]["id"] == "e3"

    def test_find_cut_point_no_cut_points(self):
        entries = []
        cut = find_cut_point(entries, 0, 0, 100)
        assert cut.first_kept_entry_index == 0

    def test_find_cut_point_split_turn(self):
        """切在 assistant 消息中间 → 视为 split turn，定位轮次起点。"""
        entries = [
            _entry(_user("a"), "e0"),
            _entry(_assistant("tool call"), "e1"),
            _entry({"role": "toolResult", "tool_call_id": "c1", "tool_name": "t",
                    "content": [{"type": "text", "text": "res"}], "is_error": False, "timestamp": 3}, "e2"),
            _entry(_user("b"), "e3"),
        ]
        # e3(1) + e2(1) = 2 < 3；e1(3) → 5 ≥ 3 → 切在 e1（assistant）
        cut = find_cut_point(entries, 0, 4, 3)
        assert cut.first_kept_entry_index == 1  # 切在 assistant 上
        assert cut.turn_start_index == 0        # 轮次起点 e0
        assert cut.is_split_turn is True


# ============================================================================
# 文件操作
# ============================================================================


class TestFileOps:
    def test_extract_and_compute(self):
        file_ops = create_file_ops()
        msg = {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "c1", "name": "read", "raw_arguments": "{}", "arguments": {"path": "a.py"}},
                {"type": "toolCall", "id": "c2", "name": "edit", "raw_arguments": "{}", "arguments": {"path": "b.py"}},
                {"type": "toolCall", "id": "c3", "name": "write", "raw_arguments": "{}", "arguments": {"path": "c.py"}},
                {"type": "toolCall", "id": "c4", "name": "read", "raw_arguments": "{}", "arguments": {"path": "c.py"}},
            ],
            "api": "openai-completions",
            "provider": "faux",
            "model": "faux-1",
            "timestamp": 1,
        }
        extract_file_ops_from_message(msg, file_ops)
        read_files, modified_files = compute_file_lists(file_ops)
        assert read_files == ["a.py"]
        assert modified_files == ["b.py", "c.py"]

    def test_extract_ignores_non_assistant(self):
        file_ops = create_file_ops()
        extract_file_ops_from_message(_user("hi"), file_ops)
        assert file_ops == {"read": set(), "written": set(), "edited": set()}

    def test_format_file_operations(self):
        assert format_file_operations([], []) == ""
        text = format_file_operations(["a.py"], ["b.py"])
        assert "<read-files>" in text
        assert "<modified-files>" in text
        assert "a.py" in text and "b.py" in text


# ============================================================================
# serialize_conversation
# ============================================================================


class TestSerializeConversation:
    def test_serialize_roles(self):
        messages = [
            _user("hello"),
            _assistant("hi there"),
            {"role": "toolResult", "tool_call_id": "c1", "tool_name": "t",
             "content": [{"type": "text", "text": "result"}], "is_error": False, "timestamp": 3},
        ]
        text = serialize_conversation(messages)
        assert "[User]: hello" in text
        assert "[Assistant]: hi there" in text
        assert "[Tool result]: result" in text

    def test_serialize_tool_result_truncated(self):
        long = "x" * 5000
        msg = {"role": "toolResult", "tool_call_id": "c1", "tool_name": "t",
               "content": [{"type": "text", "text": long}], "is_error": False, "timestamp": 3}
        text = serialize_conversation([msg])
        assert "more characters truncated" in text
        assert len(text) < 5000


# ============================================================================
# prepare_compaction
# ============================================================================


class TestPrepareCompaction:
    def test_returns_none_when_last_entry_is_compaction(self):
        entries = [
            _entry(_user("a"), "e0"),
            {"type": "compaction", "id": "c0", "parentId": "e0", "timestamp": "t",
             "summary": "s", "firstKeptEntryId": "e0", "tokensBefore": 10},
        ]
        preparation = prepare_compaction(entries, [_user("a")], DEFAULT_COMPACTION_SETTINGS)
        assert preparation is None

    def test_prepares_messages_to_summarize(self):
        entries = [
            _entry(_user("old1"), "e0"),
            _entry(_user("old2"), "e1"),
            _entry(_user("keep"), "e2"),
        ]
        context = [_user("old1"), _user("old2"), _user("keep")]
        preparation = prepare_compaction(entries, context, CompactionSettings(keep_recent_tokens=1))
        assert preparation is not None
        assert preparation.first_kept_entry_id == "e2"
        assert [m["content"] for m in preparation.messages_to_summarize] == ["old1", "old2"]
        assert preparation.is_split_turn is False

    def test_returns_none_when_nothing_to_summarize(self):
        entries = [_entry(_user("only"), "e0")]
        context = [_user("only")]
        preparation = prepare_compaction(entries, context, CompactionSettings(keep_recent_tokens=10000))
        assert preparation is None

    def test_prepare_compaction_with_previous_summary(self):
        """迭代压缩：定位上一次压缩的 summary 与边界。"""
        entries = [
            _entry(_user("old"), "e0"),
            _entry(_user("mid"), "e1"),
            {"type": "compaction", "id": "c0", "parentId": "e1", "timestamp": "t",
             "summary": "prev summary", "firstKeptEntryId": "e1", "tokensBefore": 100},
            _entry(_user("new1"), "e2"),
            _entry(_user("new2"), "e3"),
        ]
        context = [_user("new1"), _user("new2")]
        preparation = prepare_compaction(
            entries, context, CompactionSettings(keep_recent_tokens=1)
        )
        assert preparation is not None
        assert preparation.previous_summary == "prev summary"
        assert preparation.first_kept_entry_id == "e3"
        # 边界（上一次压缩 firstKeptEntryId=e1）→ 切割点 e3 之间的消息
        assert [m["content"] for m in preparation.messages_to_summarize] == ["mid", "new1"]

    def test_last_entry_compaction_returns_none(self):
        """末条已是压缩条目 → 无可压缩内容。"""
        entries = [
            _entry(_user("a"), "e0"),
            {"type": "compaction", "id": "c0", "parentId": "e0", "timestamp": "t",
             "summary": "s", "firstKeptEntryId": "e0", "tokensBefore": 10},
        ]
        assert prepare_compaction(entries, [_user("a")], DEFAULT_COMPACTION_SETTINGS) is None


# ============================================================================
# generate_summary_with_usage（LLM 摘要）
# ============================================================================


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_generates_summary(self, faux_env):
        models, core = faux_env
        core.set_responses([faux_assistant_message("## Goal\nSummarized")])

        model = models.get_model("faux", "faux-1")
        assert model is not None
        text, usage = await generate_summary_with_usage(
            [_user("hello")],
            model,
            16384,
            stream_fn=models.stream,
        )
        assert text == "## Goal\nSummarized"
        assert usage is not None
        assert usage.get("total_tokens", 0) >= 0

    @pytest.mark.asyncio
    async def test_summarization_error_raises(self, faux_env):
        models, core = faux_env
        core.set_responses([
            faux_assistant_message([], stop_reason="error", error_message="boom")
        ])
        model = models.get_model("faux", "faux-1")
        assert model is not None
        with pytest.raises(RuntimeError, match="Summarization failed"):
            await generate_summary_with_usage(
                [_user("hello")], model, 16384, stream_fn=models.stream
            )

    @pytest.mark.asyncio
    async def test_compact_split_turn(self, faux_env):
        """split turn：历史摘要 + 轮次前缀摘要两次 LLM 调用，usage 合并。"""
        models, core = faux_env
        core.set_responses([
            faux_assistant_message("## Goal\nhistory summary"),
            faux_assistant_message("prefix summary"),
        ])
        model = models.get_model("faux", "faux-1")
        assert model is not None

        entries = [
            _entry(_user("old"), "e0"),
            _entry(_assistant("old asst"), "e1"),
            _entry(_user("start turn"), "e2"),
            _entry(_assistant("tool call"), "e3"),
            _entry({"role": "toolResult", "tool_call_id": "c1", "tool_name": "t",
                    "content": [{"type": "text", "text": "res"}], "is_error": False, "timestamp": 3}, "e4"),
            _entry(_user("new"), "e5"),
        ]
        context = [_user("old"), _assistant("old asst"), _user("start turn"),
                   _assistant("tool call"), _user("new")]
        # e5(1) + e4(1) = 2 < 3；e3(3) → 5 ≥ 3 → 切在 e3（assistant）→ split turn
        preparation = prepare_compaction(
            entries, context, CompactionSettings(keep_recent_tokens=3)
        )
        assert preparation is not None
        assert preparation.is_split_turn is True

        result = await compact(preparation, model, stream_fn=models.stream)
        assert "history summary" in result.summary
        assert "Turn Context (split turn)" in result.summary
        assert "prefix summary" in result.summary
        assert core.call_count == 2
        assert result.first_kept_entry_id == "e3"
        # usage 合并
        assert result.usage is not None
        assert result.usage.get("input", 0) >= 0
