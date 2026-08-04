"""pi_ai.types.common — 跨领域基础类型。

存放不依赖其它类型模块的"叶子"定义：
- 标准化枚举（ThinkingLevel / StopReason / Transport ...）
- 简单别名（ProviderEnv / GrammarVariants ...）
- 通用协议（AsyncHTTPClient）
- 工具函数（now_ms）
"""

import time

from typing import Any, Literal, Protocol, TypedDict

# =========================================================
# 标准化枚举类型
# =========================================================

# 标准化思考深度
ThinkingLevel = Literal[
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

# 模型级思考开关（"off" 表示关闭）
ModelThinkingLevel = Literal["off"] | ThinkingLevel

# 流式响应终止原因
StopReason = Literal[
    "pending",
    "stop",
    "length",
    "tool_call",
    "error",
    "aborted",
]

# 传输协议选择
Transport = Literal[
    "sse",
    "websocket",
    "websocket-cached",
    "auto",
]

# 提示缓存保留策略
CacheRetention = Literal[
    "none",
    "short",
    "long",
]

# =========================================================
# 基础字面量与简单别名
# =========================================================

# 思考级别映射：pi 级别 -> provider 级别值；None 表示该级别不支持
ThinkingLevelMap = dict[ModelThinkingLevel, str | None]


# 各思考级别的 token 预算（仅 token-based provider）
class ThinkingBudgets(TypedDict, total=False):
    minimal: int
    low: int
    medium: int
    high: int


# chat_template_kwargs 的值（qwen/自定义模板 provider 用）
#
# 非标识符键 "$var" 需用函数式 TypedDict 语法（PEP 589）。
ChatTemplateKwargVar = TypedDict(
    "ChatTemplateKwargVar",
    {
        "$var": Literal["thinking.enabled", "thinking.effort"],
        "omitWhenOff": bool,
    },
    total=False,
)

ChatTemplateKwargValue = str | int | bool | None | ChatTemplateKwargVar

# Provider 作用域环境变量覆盖（优先于 os.environ）
ProviderEnv = dict[str, str]

# 自定义 HTTP 头；None 表示抑制默认头
ProviderHeaders = dict[str, str | None]


class AsyncHTTPClient(Protocol):
    """可注入的异步 HTTP 客户端协议（对齐 Python 生态）。

    与 httpx.AsyncClient / aiohttp.ClientSession 兼容：
    实现 `request(method, url, **kwargs)`，kwargs 透传
    （headers / json / content / timeout 等）。
    """

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any: ...


# Session 亲和性头格式
SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]


class ProviderResponse(TypedDict):
    status: int
    headers: dict[str, str]


# OpenAI grammar 变体（受约束采样）
GrammarFormat = Literal["openai_lark", "openai_regex"]
GrammarVariants = dict[GrammarFormat, str]


class JsonSchemaSampling(TypedDict):
    type: Literal["json_schema"]
    strict: Literal["prefer", "require"]


class GrammarSampling(TypedDict):
    type: Literal["grammar"]
    variants: GrammarVariants


ConstrainedSamplingConfig = JsonSchemaSampling | GrammarSampling


# =========================================================
# 工具函数
# =========================================================


def now_ms() -> int:
    """当前 Unix 时间戳（毫秒）。

    用于构造 Message / AssistantImages 的 timestamp 字段。
    """
    return int(time.time() * 1000)


__all__ = [
    "ThinkingLevel",
    "ModelThinkingLevel",
    "StopReason",
    "Transport",
    "CacheRetention",
    "ThinkingLevelMap",
    "ThinkingBudgets",
    "ChatTemplateKwargVar",
    "ChatTemplateKwargValue",
    "ProviderEnv",
    "ProviderHeaders",
    "AsyncHTTPClient",
    "SessionAffinityFormat",
    "ProviderResponse",
    "GrammarFormat",
    "GrammarVariants",
    "JsonSchemaSampling",
    "GrammarSampling",
    "ConstrainedSamplingConfig",
    "now_ms",
]
