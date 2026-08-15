"""pi_ai.utils.estimate 单元测试。

用例移植自 TS packages/ai/test/context-estimate.test.ts 与
deferred-tools.test.ts（addedToolNames 后置工具估算）。
"""

from __future__ import annotations

from pi_ai._types import Context, Model, Tool, Usage
from pi_ai.api.simple_options import clamp_max_tokens_to_context
from pi_ai.utils.estimate import (
    ContextUsageEstimate,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_and_image_content_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
)


def _usage(total_tokens: int) -> Usage:
    return {
        "input": total_tokens,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": total_tokens,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def _assistant(timestamp: int, total_tokens: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "kept"}],
        "api": "openai-responses",
        "provider": "openai",
        "model": "test-model",
        "usage": _usage(total_tokens),
        "stop_reason": "stop",
        "timestamp": timestamp,
    }


def _model(context_window: int = 10_000) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-responses",
        provider="openai",
        input=["text"],
        max_tokens=8_000,
        context_window=context_window,
    )


# ============================================================================
# calculate_context_tokens
# ============================================================================


def test_calculate_context_tokens_prefers_total_tokens():
    usage: Usage = {
        "input": 50,
        "output": 50,
        "cache_read": 10,
        "cache_write": 5,
        "total_tokens": 100,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }
    assert calculate_context_tokens(usage) == 100


def test_calculate_context_tokens_falls_back_to_sum_when_total_is_zero():
    usage: Usage = {
        "input": 50,
        "output": 30,
        "cache_read": 10,
        "cache_write": 5,
        "total_tokens": 0,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }
    assert calculate_context_tokens(usage) == 95


def test_calculate_context_tokens_handles_missing_fields():
    assert calculate_context_tokens({"input": 10, "output": 2}) == 12
    assert calculate_context_tokens({}) == 0


# ============================================================================
# 基础估算函数
# ============================================================================


def test_estimate_text_tokens():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2  # 向上取整


def test_estimate_text_and_image_content_tokens():
    # 字符串内容
    assert estimate_text_and_image_content_tokens("x" * 8) == 2
    # 文本块
    assert estimate_text_and_image_content_tokens([{"type": "text", "text": "x" * 8}]) == 2
    # 图片块按固定估算值（4800 字符）
    assert (
        estimate_text_and_image_content_tokens(
            [
                {"type": "text", "text": "x"},
                {"type": "image", "url": "http://x", "data": None, "mime_type": "image/png"},
            ]
        )
        == (1 + 4800 + 4 - 1) // 4
    )


def test_estimate_message_tokens_user_string():
    assert estimate_message_tokens({"role": "user", "content": "x" * 4, "timestamp": 1}) == 1


def test_estimate_message_tokens_tool_result():
    msg = {
        "role": "toolResult",
        "tool_call_id": "c1",
        "tool_name": "t",
        "content": [{"type": "text", "text": "x" * 8}],
        "is_error": False,
        "timestamp": 1,
    }
    assert estimate_message_tokens(msg) == 2


def test_estimate_message_tokens_assistant_blocks():
    # text(4 字符) + thinking(4 字符) + toolCall(name 3 + args JSON 7) = 18 字符 → ceil(18/4)=5
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "abcd"},
            {"type": "thinking", "thinking": "wxyz"},
            {
                "type": "toolCall",
                "id": "c1",
                "name": "foo",
                "raw_arguments": "{}",
                "arguments": {"a": 1},
            },
        ],
        "api": "responses",
        "provider": "openai",
        "model": "m",
        "timestamp": 1,
    }
    assert estimate_message_tokens(msg) == 5


def test_estimate_message_tokens_system_string():
    assert estimate_message_tokens({"role": "system", "content": "x" * 4}) == 1


def test_estimate_message_tokens_unserializable_arguments():
    # 不可序列化参数（set）回退 "[unserializable]"（16 字符）：ceil((3+16)/4)=5
    msg = {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "id": "c1",
                "name": "foo",
                "raw_arguments": "{}",
                "arguments": {"bad": {1, 2}},
            },
        ],
        "api": "responses",
        "provider": "openai",
        "model": "m",
        "timestamp": 1,
    }
    assert estimate_message_tokens(msg) == 5


def test_estimate_tools_tokens_empty_and_non_empty():
    assert estimate_tools_tokens([]) == 0
    assert estimate_tools_tokens(None) == 0
    tool = Tool(name="get_weather", description="Get weather", input_schema={})
    # {"name": "get_weather", "description": "Get weather", "input_schema": {}} = 73 字符 → ceil(73/4)=19
    assert estimate_tools_tokens([tool]) == 19


def test_clamp_max_tokens_without_context_window():
    """模型未声明 context_window（<=0）时不收敛，原样返回（含下限保护）。"""
    model = Model(
        id="m",
        provider="p",
        api="openai-completions",
        name="m",
        input=["text"],
        output=["text"],
        context_window=0,
        max_tokens=8_000,
    )
    context = Context(messages=[{"role": "user", "content": "hi"}])
    assert clamp_max_tokens_to_context(model, context, 8_000) == 8_000
    assert clamp_max_tokens_to_context(model, context, 0) == 1  # MIN_MAX_TOKENS 下限


def test_clamp_max_tokens_floor_when_context_full():
    """上下文接近窗口时 available 为负，仍保留 MIN_MAX_TOKENS 下限。"""
    model = Model(
        id="m",
        provider="p",
        api="openai-completions",
        name="m",
        input=["text"],
        output=["text"],
        context_window=100,
        max_tokens=8_000,
    )
    # 消息估算 96 tokens → available = 100 - 96 - 4096 < 0 → 收敛到 1
    context = Context(messages=[{"role": "user", "content": "x" * 384}])
    assert clamp_max_tokens_to_context(model, context, 8_000) == 1


# ============================================================================
# estimate_context_tokens — usage 语义
# ============================================================================


def test_ignores_stale_assistant_usage_after_newer_message_inserted():
    """usage 之后若插入了更新的前缀消息（如 compaction 摘要），旧 usage 失效。"""
    context = Context(
        system_prompt="system",
        messages=[
            {"role": "user", "content": "summary", "timestamp": 200},
            _assistant(100, 9_500),
            {"role": "user", "content": "x" * 4_000, "timestamp": 300},
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=1_005,
        usage_tokens=0,
        trailing_tokens=1_005,
        last_usage_index=None,
    )
    assert clamp_max_tokens_to_context(_model(), context, 8_000) == 4_899


def test_uses_assistant_usage_again_after_response_to_inserted_context():
    context = Context(
        messages=[
            {"role": "user", "content": "summary", "timestamp": 200},
            _assistant(100, 9_500),
            {"role": "user", "content": "new prompt", "timestamp": 300},
            _assistant(400, 2_000),
            {"role": "user", "content": "tail", "timestamp": 500},
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=2_001,
        usage_tokens=2_000,
        trailing_tokens=1,
        last_usage_index=3,
    )


def test_skips_aborted_and_error_usage():
    aborted = _assistant(100, 9_500)
    aborted["stop_reason"] = "aborted"
    error = _assistant(200, 9_500)
    error["stop_reason"] = "error"

    context = Context(
        messages=[aborted, error, {"role": "user", "content": "tail", "timestamp": 300}]
    )
    estimate = estimate_context_tokens(context)
    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0


def test_estimates_message_list_directly():
    messages = [
        _assistant(100, 2_000),
        {"role": "user", "content": "tail", "timestamp": 200},
    ]
    assert estimate_context_tokens(messages) == ContextUsageEstimate(
        tokens=2_001,
        usage_tokens=2_000,
        trailing_tokens=1,
        last_usage_index=0,
    )


def test_counts_tool_definitions_marked_after_latest_usage_checkpoint():
    """usage 之后 toolResult 声明的 added_tool_names 工具额外计入尾部。"""
    assistant = _assistant(100, 100)
    user_msg = {"role": "user", "content": "x" * 4, "timestamp": 200}

    plain = estimate_context_tokens(Context(messages=[assistant, user_msg], tools=[]))

    late_tool = Tool(name="late_tool", description="x" * 4_000, input_schema={})
    tool_result = {
        "role": "toolResult",
        "tool_call_id": "c1",
        "tool_name": "late_tool",
        "content": [{"type": "text", "text": "ok"}],
        "is_error": False,
        "timestamp": 300,
        "added_tool_names": ["late_tool"],
    }
    marked = estimate_context_tokens(Context(messages=[assistant, tool_result], tools=[late_tool]))

    assert marked.tokens > plain.tokens + 500
    assert marked.trailing_tokens > plain.trailing_tokens + 500
    assert marked.usage_tokens == plain.usage_tokens == 100
    assert marked.last_usage_index == plain.last_usage_index == 0


def test_added_tool_names_unknown_tool_not_counted():
    """added_tool_names 指向不在 context.tools 中的工具时不额外计入。"""
    assistant = _assistant(100, 100)
    tool_result = {
        "role": "toolResult",
        "tool_call_id": "c1",
        "tool_name": "ghost",
        "content": [{"type": "text", "text": "ok"}],
        "is_error": False,
        "timestamp": 200,
        "added_tool_names": ["ghost"],
    }
    estimate = estimate_context_tokens(Context(messages=[assistant, tool_result], tools=[]))
    assert estimate.tokens == 100 + 1  # usage 100 + toolResult "ok"(1)


def test_calculate_context_tokens_tolerates_bad_fields():
    """usage 字段为 None/字符串（外部构造消息）时不得抛 TypeError。"""
    from pi_ai.utils.estimate import calculate_context_tokens

    assert calculate_context_tokens({"input": None, "output": "5", "total_tokens": None}) == 5


def test_estimate_message_tokens_missing_content():
    """缺失/None content 的消息按空内容估算，不得崩溃。"""
    from pi_ai.utils.estimate import estimate_message_tokens

    assert estimate_message_tokens({"role": "user"}) == 0
    assert estimate_message_tokens({"role": "user", "content": None}) == 0
    assert estimate_message_tokens({"role": "toolResult", "content": None}) == 0


def test_estimate_context_tokens_string_timestamp():
    """字符串 timestamp 不得导致比较 TypeError。"""
    from pi_ai.utils.estimate import estimate_context_tokens

    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "api": "x",
            "provider": "x",
            "model": "x",
            "timestamp": "abc",
            "stop_reason": "stop",
            "usage": {
                "input": 5,
                "output": 1,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": 6,
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
            },
        }
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens >= 0
