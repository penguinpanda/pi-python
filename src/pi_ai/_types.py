"""
pi-ai 的核心类型定义（Core Types）

本模块负责定义整个 SDK 中所有公共数据结构。

主要使用两种方式：

1. TypedDict
------------------

用于描述"运行时数据（Runtime Data）"

例如：

    用户消息
    AI 回复
    流事件(Event)
    Content Block

这些对象最终都会：

- 转成 JSON
- 发给模型 API
- 从模型 API 返回

因此它们天然就是字典(dict)。

例如：

{
    "role": "user",
    "content": "Hello"
}

所以使用 TypedDict 最合适。


2. dataclass
------------------

用于描述"Python 内部对象"

例如：

- Model
- Tool
- Context

这些对象通常：

- 在 Python 内部创建
- 保存配置
- 不直接来自网络

因此使用 dataclass 更方便。

=========================================================
整体关系
=========================================================

               Context
                   │
        ┌──────────┴──────────┐
        │                     │
    Message[]              Tool[]
        │
        │
        ├──────── UserMessage
        ├──────── AssistantMessage
        ├──────── ToolResultMessage
        └──────── SystemMessage

AssistantMessage
        │
        ├──── ContentBlock[]
        │        ├── TextContent
        │        ├── ImageContent
        │        ├── ThinkingContent
        │        └── ToolCallContent
        │
        └──── Usage

=========================================================
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Union, NotRequired

# =========================================================
# Content Block（消息内容块）
# =========================================================
#
# 一个消息(Message)并不是只能包含文本。
#
# 例如：
#
# 用户消息：
#
# [
#     {"type":"text", ...},
#     {"type":"image", ...}
# ]
#
# Assistant 消息：
#
# [
#     {"type":"thinking", ...},
#     {"type":"text", ...},
#     {"type":"toolCall", ...}
# ]
#
# 因此统一抽象为 ContentBlock。
# =========================================================


class TextContent(TypedDict):
    """
    文本内容块

    示例：

    {
        "type": "text",
        "text": "Hello"
    }
    """

    type: Literal["text"]
    text: str # 文本内容


class ImageContent(TypedDict):
    """
    图片内容块

    可以使用：

    ① 图片 URL

    {
        "type":"image",
        "url":"https://..."
    }

    ② Base64 数据

    {
        "type":"image",
        "data":"xxxxx"
    }
    """

    type: Literal["image"]
    url: str | None       # 图片地址
    data: str | None      # Base64 编码图片
    mediaType: str | None # MIME 类型，例如 image/png


class ToolCallContent(TypedDict):
    """
    Tool Calling 内容块

    AI 告诉客户端：

    "我要调用某个工具。"

    示例：

    {
        "type":"toolCall",
        "toolName":"search",
        "args":"{...}"
    }
    """

    type: Literal["toolCall"]
    toolCallId: str # Tool Call 唯一 ID
    toolName: str   # 工具名称
    args: str       # JSON 字符串形式参数


class ThinkingContent(TypedDict):
    """
    推理内容块（Reasoning）

    部分模型（如 DeepSeek、Claude）
    会返回中间思考过程。
    """

    type: Literal["thinking"]
    thinking: str           # 思考内容
    signature: str | None   # 推理签名（某些 API 使用）


# ContentBlock 可以是四种内容中的任意一种
ContentBlock = Union[
    TextContent,
    ImageContent,
    ToolCallContent,
    ThinkingContent,
]

# =========================================================
# Message（消息）
# =========================================================
#
# 对话历史全部由 Message 组成。
#
# Context.messages
#
# [
#     SystemMessage,
#     UserMessage,
#     AssistantMessage,
#     ToolResultMessage
# ]
#
# =========================================================

# =========================================================
# 标准化枚举类型（与 TS types.ts 对齐）
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
    "toolUse",
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


class SystemMessage(TypedDict):
    """System Prompt"""

    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    """
    用户消息

    可以：

    ① 纯文本

        "你好"

    ② 多模态

        [
            TextContent,
            ImageContent
        ]
    """

    role: Literal["user"]
    content: str | list[Union[TextContent, ImageContent]]


class _AssistantMessageBase(TypedDict):
    """
    AssistantMessage 公共字段
    """

    role: Literal["assistant"]
    content: list[ContentBlock]             # 输出内容
    api: str                                # 使用哪个 API
    provider: str                           # Provider 名称
    model: str                              # 模型名称
    stopReason: NotRequired[StopReason]     # 停止原因
    errorMessage: NotRequired[str | None]   # 错误信息
    timestamp: NotRequired[int]             # 时间戳


class Usage(TypedDict, total=False):
    """
    Token 使用统计

    total=False 表示全部字段可选。
    """

    input: int
    output: int
    cacheRead: int
    cacheWrite: int
    totalTokens: int

    cost: NotRequired[dict[str, float]] # 成本统计


class AssistantMessage(_AssistantMessageBase):
    """
    Assistant 回复消息
    """

    usage: NotRequired[Usage]


class ToolResultMessage(TypedDict):
    """
    Tool 返回结果

    工具执行完成以后，
    会将结果作为新的 Message
    再发给模型。
    """

    role: Literal["toolResult"]
    toolCallId: str
    toolName: str
    content: list[Union[TextContent, ImageContent]]
    addedToolNames: NotRequired[list[str] | None]


# 所有 Message 的联合类型
Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
]

# =========================================================
# Tool（工具定义）
# =========================================================


@dataclass(slots=True)
class Tool:
    """
    一个可供模型调用的工具。
    """

    
    name: str           # 工具名称
    description: str    # 工具说明
    inputSchema: dict   # JSON Schema


# =========================================================
# Model（模型元数据）
# =========================================================

KnownApi = Literal[
    "openai-completions",
    "openai-responses",
]

KnownProvider = Literal[
    "openai",
    "deepseek",
    "ollama",
]


@dataclass(slots=True)
class Model:
    """
    模型元数据

    用于描述：

    - 支持哪些能力
    - Token 限制
    - 是否支持图片
    """

    
    id: str             # 模型唯一 ID
    provider: str
    api: str            # API 类型
    name: str = ""      # 模型名称
    input: list[str] = field(default_factory=list)          # 输入能力
    output: list[str] = field(default_factory=list)         # 输出能力
    cost: dict[str, float] = field(default_factory=dict)    # Token 成本
    maxTokens: int = 4096               # 最大 Token
    thinking: bool = False              # 是否支持 Thinking
    supportsToolCalling: bool = True    # 是否支持 Tool Calling
    supportsImages: bool = False        # 是否支持图片

    def capabilities(self) -> list[str]:
        """返回模型支持的能力标签列表（thinking / tools / images）。

        供 CLI、日志等展示层消费，避免各调用方重复拼装能力逻辑。
        """
        caps: list[str] = []
        if self.thinking:
            caps.append("thinking")
        if self.supportsToolCalling:
            caps.append("tools")
        if self.supportsImages:
            caps.append("images")
        return caps


# =========================================================
# Context
# =========================================================


@dataclass(slots=True)
class Context:
    """
    一次模型请求的完整上下文。

    最终会发送给 Provider。
    """

    # 对话历史
    messages: list[Message]

    # 工具列表
    tools: list[Tool] = field(default_factory=list)

    # System Prompt
    system: str | None = None


# =========================================================
# StreamOptions
# =========================================================


class StreamOptions(TypedDict, total=False):
    """
    流式请求参数。
    """

    temperature: float
    maxTokens: int
    apiKey: str
    thinkingBudget: int | None
    thinkingEnabled: bool | None
    headers: dict[str, str | None]

    # 最大重试次数（provider 层客户端重试上限）。
    maxRetries: int

    # 最大重试延迟（毫秒）。
    #
    # 当服务器要求等待超过该值时直接失败，
    # 交由上层重试逻辑处理。
    maxRetryDelayMs: int

    # 可选的中止信号（asyncio.Event）。
    #
    # 支持流式中止的 Provider（例如 Faux Provider）
    # 会在流式输出过程中检查它，
    # 一旦被 set 即以 aborted 结束。
    signal: NotRequired[Any]


# =========================================================
# Stream Event（流事件）
# =========================================================
#
# EventStream 会不断产生这些事件。
#
# delta
# delta
# delta
# toolCallDelta
# delta
# done
#
# =========================================================


class DeltaEvent(TypedDict):
    """文本增量"""

    type: Literal["delta"]
    text: str


class ToolCallDeltaEvent(TypedDict):
    """工具调用增量"""

    type: Literal["toolCallDelta"]
    toolCallId: str
    toolName: str
    argsDelta: str


class ThinkingDeltaEvent(TypedDict):
    """思考内容增量"""

    type: Literal["thinkingDelta"]
    thinking: str


class DoneEvent(TypedDict):
    """流结束"""

    type: Literal["done"]
    # 成功终止原因（与 TS 对齐，仅限成功终止值）
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class ErrorEvent(TypedDict):
    """流异常结束"""

    type: Literal["error"]
    # 异常终止原因（与 TS 对齐，仅限异常终止值）
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Union[
    DeltaEvent,
    ToolCallDeltaEvent,
    ThinkingDeltaEvent,
    DoneEvent,
    ErrorEvent,
]

# 暴露接口
__all__ = [
    "TextContent",
    "ImageContent",
    "ToolCallContent",
    "ThinkingContent",
    "ContentBlock",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    "Usage",
    "Tool",
    "Model",
    "KnownApi",
    "KnownProvider",
    "ThinkingLevel",
    "ModelThinkingLevel",
    "StopReason",
    "Transport",
    "CacheRetention",
    "Context",
    "StreamOptions",
    "DeltaEvent",
    "ToolCallDeltaEvent",
    "ThinkingDeltaEvent",
    "DoneEvent",
    "ErrorEvent",
    "AssistantMessageEvent",
]
