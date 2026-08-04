"""pi_ai.types.compat — 兼容配置（Compat）与路由。

Compat 字段为 provider 配置键。字段名使用 camelCase：这些键会作为
JSON 请求字段直接发送给 provider，是上游 TS 生态的序列化格式约定。

"""

from typing import Any, Literal, TypedDict

from .common import (
    ChatTemplateKwargValue,
    SessionAffinityFormat,
)


class OpenRouterRouting(TypedDict, total=False):
    """OpenRouter provider 路由偏好（作为 provider 请求字段发送）"""

    allow_fallbacks: bool
    require_parameters: bool
    data_collection: Literal["deny", "allow"]
    zdr: bool
    enforce_distillable_text: bool
    order: list[str]
    only: list[str]
    ignore: list[str]
    quantizations: list[str]
    sort: str | dict[str, Any]
    max_price: dict[str, Any]
    preferred_min_throughput: float | dict[str, Any]
    preferred_max_latency: float | dict[str, Any]


class VercelGatewayRouting(TypedDict, total=False):
    """Vercel AI Gateway 路由偏好"""

    only: list[str]
    order: list[str]


class OpenAICompletionsCompat(TypedDict, total=False):
    """OpenAI-compatible completions API 兼容设置"""

    supportsStore: bool
    supportsDeveloperRole: bool
    supportsReasoningEffort: bool
    supportsUsageInStreaming: bool
    supportsFinishReason: bool
    maxTokensField: Literal["max_completion_tokens", "max_tokens"]
    requiresToolResultName: bool
    requiresAssistantAfterToolResult: bool
    requiresThinkingAsText: bool
    requiresReasoningContentOnAssistantMessages: bool
    thinkingFormat: Literal[
        "openai",
        "openrouter",
        "deepseek",
        "together",
        "zai",
        "qwen",
        "chat-template",
        "qwen-chat-template",
        "string-thinking",
        "ant-ling",
    ]
    chatTemplateKwargs: dict[str, ChatTemplateKwargValue]
    openRouterRouting: OpenRouterRouting
    vercelGatewayRouting: VercelGatewayRouting
    zaiToolStream: bool
    supportsOpenAIGrammarTools: bool
    supportsStrictMode: bool
    cacheControlFormat: Literal["anthropic"]
    sendSessionAffinityHeaders: bool
    deferredToolsMode: Literal["kimi"]
    sessionAffinityFormat: SessionAffinityFormat
    supportsLongCacheRetention: bool


class OpenAIResponsesCompat(TypedDict, total=False):
    """OpenAI Responses API 兼容设置"""

    supportsDeveloperRole: bool
    sessionAffinityFormat: SessionAffinityFormat
    supportsLongCacheRetention: bool
    supportsStrictMode: bool
    supportsOpenAIGrammarTools: bool
    supportsToolSearch: bool
    supportsExplicitPromptCacheMode: bool


class AnthropicMessagesCompat(TypedDict, total=False):
    """Anthropic Messages 兼容设置"""

    supportsEagerToolInputStreaming: bool
    supportsLongCacheRetention: bool
    sendSessionAffinityHeaders: bool
    supportsCacheControlOnTools: bool
    supportsTemperature: bool
    forceAdaptiveThinking: bool
    allowEmptySignature: bool
    supportsStrictTools: bool
    supportsToolReferences: bool


class BedrockCompat(TypedDict, total=False):
    """Amazon Bedrock 兼容设置"""

    supportsStrictMode: bool


# 兼容配置是扩展字段；Model.compat 可为任一种 API 的兼容配置。
ModelCompat = (
    OpenAICompletionsCompat | OpenAIResponsesCompat | AnthropicMessagesCompat | BedrockCompat
)


__all__ = [
    "OpenRouterRouting",
    "VercelGatewayRouting",
    "OpenAICompletionsCompat",
    "OpenAIResponsesCompat",
    "AnthropicMessagesCompat",
    "BedrockCompat",
    "ModelCompat",
]
