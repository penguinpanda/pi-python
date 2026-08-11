"""pi_ai.utils.overflow 单元测试。

用例移植自 TS packages/ai/test/overflow.test.ts。
"""

from __future__ import annotations

from pi_ai._types import AssistantMessage
from pi_ai.utils.overflow import (
    NON_OVERFLOW_PATTERNS,
    OVERFLOW_PATTERNS,
    get_overflow_patterns,
    is_context_overflow,
)


def _error_msg(error_message: str) -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [],
        "api": "openai-completions",
        "provider": "ollama",
        "model": "qwen3.5:35b",
        "usage": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 0,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
        "stop_reason": "error",
        "error_message": error_message,
        "timestamp": 1_700_000_000_000,
    }


def _usage_msg(input_tokens: int, output: int = 0, cache_read: int = 0) -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [],
        "api": "openai-completions",
        "provider": "xiaomi",
        "model": "mimo-v2.5-pro",
        "usage": {
            "input": input_tokens,
            "output": output,
            "cache_read": cache_read,
            "cache_write": 0,
            "total_tokens": input_tokens + cache_read + output,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
        "stop_reason": "stop",
        "timestamp": 1_700_000_000_000,
    }


def _length_stop_msg(input_tokens: int, cache_read: int, output: int) -> AssistantMessage:
    msg = _usage_msg(input_tokens, output, cache_read)
    msg["stop_reason"] = "length"
    return msg


# ============================================================================
# Case 1: 错误式溢出
# ============================================================================


def test_detects_explicit_ollama_prompt_too_long():
    message = _error_msg("400 `prompt too long; exceeded max context length by 100918 tokens`")
    assert is_context_overflow(message, 32768) is True


def test_detects_together_ai_context_length_errors():
    message = _error_msg(
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens)."
    )
    assert is_context_overflow(message, 262144) is True


def test_detects_litellm_wrapped_openai_max_context_errors():
    message = _error_msg(
        "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: "
        "litellm.APIConnectionError: APIConnectionError: OpenAIException - "
        "Requested token count exceeds the model's maximum context length of 131072 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_openai_compatible_parenthesized_max_context_errors():
    message = _error_msg(
        "Error: 400 Input length (265330) exceeds model's maximum context length (262144)."
    )
    assert is_context_overflow(message, 262144) is True


def test_detects_openrouter_poolside_max_allowed_input_errors():
    message = _error_msg(
        "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_ds4_configured_context_size_errors():
    message = _error_msg(
        "400 Prompt has 256468 tokens, but the configured context size is 256000 tokens"
    )
    assert is_context_overflow(message, 256000) is True

    comma_message = _error_msg(
        "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens"
    )
    assert is_context_overflow(comma_message, 256000) is True


def test_detects_anthropic_and_google_patterns():
    assert (
        is_context_overflow(
            _error_msg("prompt is too long: 213462 tokens > 200000 maximum"), 200_000
        )
        is True
    )
    assert (
        is_context_overflow(
            _error_msg(
                '413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}'
            ),
            200_000,
        )
        is True
    )
    assert (
        is_context_overflow(
            _error_msg(
                "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"
            ),
            1_048_576,
        )
        is True
    )


def test_detects_xai_grok_pattern():
    message = _error_msg(
        "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
    )
    assert is_context_overflow(message, 131_072) is True


def test_detects_cerebras_no_body_pattern():
    assert is_context_overflow(_error_msg("400 (no body)"), 131_072) is True
    assert is_context_overflow(_error_msg("413 status code (no body)"), 131_072) is True


def test_detects_mistral_pattern():
    message = _error_msg(
        "Prompt contains 100000 tokens which is too large for model with 32000 maximum context length"
    )
    assert is_context_overflow(message, 32_000) is True


def test_detects_dashscope_range_pattern():
    message = _error_msg("Range of input length should be [1, 32000]")
    assert is_context_overflow(message, 32_000) is True


# ============================================================================
# NON_OVERFLOW 排除
# ============================================================================


def test_does_not_treat_generic_non_overflow_ollama_errors_as_overflow():
    message = _error_msg("500 `model runner crashed unexpectedly`")
    assert is_context_overflow(message, 32768) is False


def test_does_not_treat_bedrock_throttling_as_overflow():
    # Bedrock 限流也会出现 "Too many tokens"（HTTP 429），但并非上下文溢出。
    message = _error_msg("Throttling error: Too many tokens, please wait before trying again.")
    assert is_context_overflow(message, 200_000) is False


def test_does_not_treat_bedrock_service_unavailable_as_overflow():
    message = _error_msg("Service unavailable: The service is temporarily unavailable.")
    assert is_context_overflow(message, 200_000) is False


def test_does_not_treat_generic_rate_limit_errors_as_overflow():
    message = _error_msg("Rate limit exceeded, please retry after 30 seconds.")
    assert is_context_overflow(message, 200_000) is False


def test_does_not_treat_http_429_style_errors_as_overflow():
    message = _error_msg("Too many requests. Please slow down.")
    assert is_context_overflow(message, 200_000) is False


# ============================================================================
# Case 2: 静默溢出（z.ai 风格）
# ============================================================================


def test_detects_silent_overflow_via_usage_input():
    message = _usage_msg(input_tokens=100_001)
    assert is_context_overflow(message, 100_000) is True

    # 未超出窗口
    assert is_context_overflow(_usage_msg(input_tokens=50_000), 100_000) is False

    # 未提供 context_window 时不检测静默溢出
    assert is_context_overflow(_usage_msg(input_tokens=100_001)) is False


def test_silent_overflow_counts_cache_read_toward_input():
    # input + cache_read 合计超出窗口
    message = _usage_msg(input_tokens=60_000, cache_read=50_000)
    assert is_context_overflow(message, 100_000) is True


# ============================================================================
# Case 3: 截断式溢出（Xiaomi MiMo 风格）
# ============================================================================


def test_detects_xiaomi_style_length_stop_overflow():
    message = _length_stop_msg(input_tokens=58, cache_read=1_048_512, output=0)
    assert is_context_overflow(message, 1_048_576) is True


def test_does_not_treat_normal_length_stops_with_output_as_overflow():
    message = _length_stop_msg(input_tokens=1_000, cache_read=0, output=4_096)
    assert is_context_overflow(message, 200_000) is False


def test_does_not_treat_length_stops_far_below_context_as_overflow():
    message = _length_stop_msg(input_tokens=100, cache_read=0, output=0)
    assert is_context_overflow(message, 200_000) is False


def test_length_stop_requires_context_window():
    message = _length_stop_msg(input_tokens=1_000_000, cache_read=0, output=0)
    assert is_context_overflow(message) is False


# ============================================================================
# get_overflow_patterns
# ============================================================================


def test_pattern_counts():
    assert len(OVERFLOW_PATTERNS) == 25
    assert len(NON_OVERFLOW_PATTERNS) == 3


def test_get_overflow_patterns_returns_copy():
    patterns = get_overflow_patterns()
    patterns.clear()
    assert len(OVERFLOW_PATTERNS) == 25
    assert len(get_overflow_patterns()) == 25


def test_chinese_only_error_message_not_matched():
    """纯中文溢出错误当前不命中英文模式：验证不误报、不抛异常（已知限制）。"""
    msg = _error_msg("输入长度超过模型最大上下文窗口，请减少输入内容")
    assert is_context_overflow(msg, 8192) is False


def test_english_pattern_matches_within_chinese_context():
    """英文模式嵌入中英混合消息中仍能命中。"""
    msg = _error_msg("请求失败：prompt is too long，请缩短输入")
    assert is_context_overflow(msg, 8192) is True
