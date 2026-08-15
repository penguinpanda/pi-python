"""Model.compat 运行时逻辑（对齐 TS 各 api/*.ts 对 compat 的消费）。

types/compat.py 只定义了 Compat 配置类型；本模块把配置键转化为
请求构造阶段的运行时决策（maxTokens 字段名、角色支持、thinking 格式、
strict/grammar 开关、缓存语义等）。
"""

from typing import Any, cast

from ..types import Model


def compat_value(model: Model, key: str, default: Any = None) -> Any:
    """读取模型 compat 配置；未配置时返回 default。"""
    compat = model.compat
    return compat.get(key, default) if compat else default


def detect_completions_compat(model: Model) -> dict[str, Any]:
    """按 provider/base_url 自动检测 OpenAI-compatible completions 兼容性。

    移植 TS openai-completions.ts detectCompat。显式 ``model.compat`` 字段
    逐键覆盖检测值。该函数只用于 openai-completions；其它 API 的默认值
    不应被此处的 provider 启发式改变。
    """
    provider = model.provider
    base_url = model.base_url or ""
    lower_url = base_url.lower()

    is_zai = (
        provider == "zai"
        or provider == "zai-coding-cn"
        or "api.z.ai" in lower_url
        or "open.bigmodel.cn" in lower_url
    )
    is_together = (
        provider == "together" or "api.together.ai" in lower_url or "api.together.xyz" in lower_url
    )
    is_moonshot = (
        provider == "moonshotai" or provider == "moonshotai-cn" or "api.moonshot." in lower_url
    )
    is_openrouter = provider == "openrouter" or "openrouter.ai" in lower_url
    is_cloudflare_workers = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in lower_url
    is_cloudflare_gateway = (
        provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in lower_url
    )
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in lower_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in lower_url
    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in lower_url
        or provider == "xai"
        or "api.x.ai" in lower_url
        or is_together
        or "chutes.ai" in lower_url
        or "deepseek.com" in lower_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in lower_url
        or is_cloudflare_workers
        or is_cloudflare_gateway
        or is_ant_ling
    )
    use_max_tokens = (
        "chutes.ai" in lower_url
        or is_moonshot
        or is_cloudflare_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )
    is_grok = provider == "xai" or "api.x.ai" in lower_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in lower_url
    is_openrouter_developer_role_model = is_openrouter and (
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    detected: dict[str, Any] = {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_openrouter_developer_role_model
        or (not is_non_standard and not is_openrouter),
        "supportsReasoningEffort": not any(
            (
                is_grok,
                is_zai,
                is_moonshot,
                is_together,
                is_cloudflare_gateway,
                is_nvidia,
                is_ant_ling,
            )
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": (
            "deepseek"
            if is_deepseek
            else "zai"
            if is_zai
            else "together"
            if is_together
            else "ant-ling"
            if is_ant_ling
            else "openrouter"
            if is_openrouter
            else "openai"
        ),
        "zaiToolStream": False,
        "supportsThinkingTokenBudget": False,
        "supportsStrictMode": not any((is_moonshot, is_together, is_cloudflare_gateway, is_nvidia)),
        "supportsOpenAIGrammarTools": False,
        "cacheControlFormat": (
            "anthropic" if is_openrouter and model.id.startswith("anthropic/") else None
        ),
        "sendSessionAffinityHeaders": False,
        "deferredToolsMode": None,
        "sessionAffinityFormat": "openrouter" if is_openrouter else "openai",
        "supportsLongCacheRetention": not any(
            (
                is_together,
                is_cloudflare_workers,
                is_cloudflare_gateway,
                is_nvidia,
                is_ant_ling,
            )
        ),
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
    }
    if model.compat:
        compat = cast(dict[str, Any], model.compat)
        for key in list(detected.keys()):
            value = compat.get(key)
            if value is not None:
                detected[key] = value
    return detected


def completions_compat_value(model: Model, key: str, default: Any = None) -> Any:
    """openai-completions 专用的 compat 读取：显式配置 > provider 自动检测。"""
    if model.compat:
        compat = cast(dict[str, Any], model.compat)
        if key in compat and compat.get(key) is not None:
            return compat[key]
    return detect_completions_compat(model).get(key, default)


def max_tokens_field(model: Model) -> str:
    """maxTokens 请求字段名。

    completions 走 provider/URL 自动检测；生成目录显式配置优先。
    """
    if model.api == "openai-completions":
        return str(completions_compat_value(model, "maxTokensField", "max_completion_tokens"))
    return str(compat_value(model, "maxTokensField", "max_tokens"))


def supports_developer_role(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "supportsDeveloperRole", True))
    return bool(compat_value(model, "supportsDeveloperRole", True))


def supports_reasoning_effort(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "supportsReasoningEffort", True))
    return bool(compat_value(model, "supportsReasoningEffort", True))


def requires_reasoning_content_on_assistant_messages(model: Model) -> bool:
    """DeepSeek 等要求历史 assistant 消息携带 reasoning_content。"""
    if model.api == "openai-completions":
        return bool(
            completions_compat_value(model, "requiresReasoningContentOnAssistantMessages", False)
        )
    return bool(compat_value(model, "requiresReasoningContentOnAssistantMessages", False))


def thinking_format(model: Model) -> str:
    """thinking 编码格式（openai / openrouter / deepseek / together / zai / qwen...）。"""
    if model.api == "openai-completions":
        return str(completions_compat_value(model, "thinkingFormat", "openai"))
    return str(compat_value(model, "thinkingFormat", "openai"))


def supports_strict_mode(model: Model) -> bool:
    """各 API 的 strict-tool 默认值不同。

    openai-responses / openai-codex-responses 在 TS 侧默认 false；
    azure-openai-responses 与 openai-completions 默认 true。
    """
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "supportsStrictMode", True))
    if model.api in ("openai-responses", "openai-codex-responses"):
        return bool(compat_value(model, "supportsStrictMode", False))
    return bool(compat_value(model, "supportsStrictMode", True))


def supports_openai_grammar_tools(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "supportsOpenAIGrammarTools", False))
    return bool(compat_value(model, "supportsOpenAIGrammarTools", False))


def supports_long_cache_retention(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "supportsLongCacheRetention", True))
    return bool(compat_value(model, "supportsLongCacheRetention", True))


def requires_assistant_after_tool_result(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "requiresAssistantAfterToolResult", False))
    return bool(compat_value(model, "requiresAssistantAfterToolResult", False))


def requires_tool_result_name(model: Model) -> bool:
    if model.api == "openai-completions":
        return bool(completions_compat_value(model, "requiresToolResultName", False))
    return bool(compat_value(model, "requiresToolResultName", False))


__all__ = [
    "compat_value",
    "detect_completions_compat",
    "completions_compat_value",
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
