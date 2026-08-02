"""pi_ai.utils.overflow — Context 溢出检测（移植 TS packages/ai/src/utils/overflow.ts）。

三类溢出路径：

1. 错误式溢出：多数 provider 以 stop_reason="error" + 特征错误消息返回
2. 静默溢出（z.ai 风格）：接受溢出请求但成功返回，
   靠 usage.input + cache_read > context_window 识别
3. 截断式溢出（Xiaomi MiMo 风格）：服务端把超长输入截到正好填满
   context_window，返回 finish_reason="length" 且 output=0
"""

from __future__ import annotations

import re

from .._types import AssistantMessage

# ------------------------------------------------------------------
# 各 provider 的上下文溢出错误模式。
#
# 这些模式匹配输入超过模型上下文窗口时返回的错误消息。
# （注释中给出了各 provider 的示例错误消息）
# ------------------------------------------------------------------
OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),  # Anthropic token overflow
    re.compile(r"request_too_large", re.IGNORECASE),  # Anthropic request byte-size overflow (HTTP 413)
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Amazon Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),  # OpenAI (Completions & Responses API)
    # OpenAI-compatible proxies (LiteLLM)
    re.compile(
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        re.IGNORECASE,
    ),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Google (Gemini)
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI (Grok)
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter (most backends)
    # OpenRouter/Poolside
    re.compile(
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        re.IGNORECASE,
    ),
    # Together AI
    re.compile(
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        re.IGNORECASE,
    ),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp server
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),  # Kimi For Coding
    # Mistral
    re.compile(
        r"too large for model with \d+ maximum context length",
        re.IGNORECASE,
    ),
    # DS4 server
    re.compile(
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        re.IGNORECASE,
    ),
    # z.ai non-standard finish_reason surfaced as error text
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),
    # Ollama explicit overflow error
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),
    # DashScope / Qwen Token Plan
    re.compile(r"range of input length should be", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"too many tokens", re.IGNORECASE),  # Generic fallback
    re.compile(r"token limit exceeded", re.IGNORECASE),  # Generic fallback
    # Cerebras: 400/413 with no body
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),
]

# ------------------------------------------------------------------
# 非溢出错误模式。
#
# 即使同时匹配 OVERFLOW_PATTERNS，命中以下任一模式也不视为溢出
# （例如 Bedrock 的限流错误 "Throttling error: Too many tokens..."，
# 会命中 /too many tokens/i 溢出模式，但实际是限流）。
# ------------------------------------------------------------------
NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    # AWS Bedrock 非溢出错误（formatBedrockError 的人类可读前缀）
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),  # Generic rate limiting
    re.compile(r"too many requests", re.IGNORECASE),  # Generic HTTP 429 style
]


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """判断一条 assistant 消息是否表示上下文溢出。

    Case 1：多数 provider 以 stop_reason="error" + 特征错误消息返回；
            命中 NON_OVERFLOW_PATTERNS 时排除。
    Case 2：静默溢出（z.ai 风格）——成功返回但 usage.input+cache_read
            超过 context_window。
    Case 3：截断式溢出（Xiaomi MiMo 风格）——stop_reason="length"、
            output=0 且 input+cache_read 填满 context_window（≥99%）。
    """
    # Case 1: 错误消息模式。
    if message.get("stop_reason") == "error":
        error_message = message.get("error_message")
        if error_message:
            is_non_overflow = any(
                pattern.search(error_message) for pattern in NON_OVERFLOW_PATTERNS
            )
            if not is_non_overflow and any(
                pattern.search(error_message) for pattern in OVERFLOW_PATTERNS
            ):
                return True

    usage = message.get("usage") or {}
    stop_reason = message.get("stop_reason")

    # Case 2: 静默溢出（z.ai 风格）——成功但 usage 超出上下文窗口。
    if context_window and stop_reason == "stop":
        input_tokens = usage.get("input", 0) + usage.get("cache_read", 0)
        if input_tokens > context_window:
            return True

    # Case 3: 截断式溢出（Xiaomi MiMo 风格）——length 停止且无输出、
    #         输入填满上下文窗口。
    if context_window and stop_reason == "length" and usage.get("output", 0) == 0:
        input_tokens = usage.get("input", 0) + usage.get("cache_read", 0)
        if input_tokens >= context_window * 0.99:
            return True

    return False


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """返回溢出模式列表（供测试使用）。"""
    return list(OVERFLOW_PATTERNS)


__all__ = [
    "NON_OVERFLOW_PATTERNS",
    "OVERFLOW_PATTERNS",
    "get_overflow_patterns",
    "is_context_overflow",
]
