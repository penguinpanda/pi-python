"""Model.compat 运行时逻辑（对齐 TS 各 api/*.ts 对 compat 的消费）。

types/compat.py 只定义了 Compat 配置类型；本模块把配置键转化为
请求构造阶段的运行时决策（maxTokens 字段名、角色支持、thinking 格式、
strict/grammar 开关、缓存语义等）。
"""

from typing import Any

from ..types import Model


def compat_value(model: Model, key: str, default: Any = None) -> Any:
    """读取模型 compat 配置；未配置时返回 default。"""
    compat = model.compat
    return compat.get(key, default) if compat else default


def max_tokens_field(model: Model) -> str:
    """maxTokens 请求字段名：max_tokens（Python 既有默认）或 compat 指定值。

    生成目录中的模型会携带 TS 产出的 maxTokensField
    （OpenAI 为 max_completion_tokens，DeepSeek 等为 max_tokens），
    显式配置时按配置切换；无配置保持既有行为。
    """
    return compat_value(model, "maxTokensField", "max_tokens")


def supports_developer_role(model: Model) -> bool:
    return bool(compat_value(model, "supportsDeveloperRole", True))


def supports_reasoning_effort(model: Model) -> bool:
    return bool(compat_value(model, "supportsReasoningEffort", True))


def requires_reasoning_content_on_assistant_messages(model: Model) -> bool:
    """DeepSeek 等要求历史 assistant 消息携带 reasoning_content。"""
    return bool(compat_value(model, "requiresReasoningContentOnAssistantMessages", False))


def thinking_format(model: Model) -> str:
    """thinking 编码格式（openai / openrouter / deepseek / together / zai / qwen...）。"""
    return compat_value(model, "thinkingFormat", "openai")


def supports_strict_mode(model: Model) -> bool:
    return bool(compat_value(model, "supportsStrictMode", True))


def supports_openai_grammar_tools(model: Model) -> bool:
    return bool(compat_value(model, "supportsOpenAIGrammarTools", False))


def supports_long_cache_retention(model: Model) -> bool:
    return bool(compat_value(model, "supportsLongCacheRetention", True))


def requires_assistant_after_tool_result(model: Model) -> bool:
    return bool(compat_value(model, "requiresAssistantAfterToolResult", False))


def requires_tool_result_name(model: Model) -> bool:
    return bool(compat_value(model, "requiresToolResultName", False))


__all__ = [
    "compat_value",
    "max_tokens_field",
    "supports_developer_role",
    "supports_reasoning_effort",
    "requires_reasoning_content_on_assistant_messages",
    "thinking_format",
    "supports_strict_mode",
    "supports_openai_grammar_tools",
    "supports_long_cache_retention",
    "requires_assistant_after_tool_result",
    "requires_tool_result_name",
]
