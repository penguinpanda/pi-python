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
        │        └── ToolCall
        │
        └──── Usage

=========================================================
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from ._event_stream import AssistantMessageEventStream

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

    # 新增：OpenAI responses 的 message metadata（旧版 id 字符串或 TextSignatureV1 JSON）
    textSignature: NotRequired[str]


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
    url: str | None       # 图片地址（Python 扩展：TS 无 url 字段）
    data: str | None      # Base64 编码图片
    mimeType: str | None  # MIME 类型，例如 image/png（对齐 TS）


class ToolCall(TypedDict):
    """
    Tool Calling 内容块

    AI 告诉客户端：

    "我要调用某个工具。"

    示例：

    {
        "type":"toolCall",
        "id":"call_1",
        "name":"search",
        "arguments":{"query":"..."}
    }
    """

    type: Literal["toolCall"]
    id: str                           # Tool Call 唯一 ID
    name: str                         # 工具名称
    arguments: dict[str, Any]         # 已解析的 JSON 对象（流式累积原始字符串后解析）
    thoughtSignature: NotRequired[str]  # Google 专用：复用思考上下文的签名


class ThinkingContent(TypedDict):
    """
    推理内容块（Reasoning）

    部分模型（如 DeepSeek、Claude）
    会返回中间思考过程。
    """

    type: Literal["thinking"]
    thinking: str                       # 思考内容
    thinkingSignature: NotRequired[str] # 推理签名（某些 API 使用；如 OpenAI responses 的 reasoning item ID）

    # True 表示思考内容被安全过滤改写（加密占位存于 thinkingSignature，以支持多轮续传）
    redacted: NotRequired[bool]


# ContentBlock 可以是四种内容中的任意一种
ContentBlock = Union[
    TextContent,
    ImageContent,
    ToolCall,
    ThinkingContent,
]


class TextSignatureV1(TypedDict):
    """OpenAI responses 的文本签名（v1 结构）"""

    v: Literal[1]
    id: str
    phase: NotRequired[Literal["commentary", "final_answer"]]

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

# =========================================================
# 基础字面量与简单别名（对齐 TS types.ts）
# =========================================================

# 思考级别映射：pi 级别 -> provider 级别值；None 表示该级别不支持
# TS: Partial<Record<ModelThinkingLevel, string | null>>
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

# TS FetchFunction；Python 无 fetch，保留字段用于对齐，
# 实现时可映射为 httpx.AsyncClient 或直接忽略（标注为扩展）
FetchFunction = Callable[..., Any]

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
    timestamp: int  # Unix 毫秒时间戳（必填，对齐 TS）


class Cost(TypedDict):
    """成本统计（$；对齐 TS Usage.cost）"""

    input: float
    output: float
    cacheRead: float
    cacheWrite: float
    total: float


class Usage(TypedDict):
    """
    Token 使用统计（对齐 TS Usage；字段必填）
    """

    input: int
    output: int
    cacheRead: int
    cacheWrite: int
    totalTokens: int
    cost: Cost

    # 可选字段
    cacheWrite1h: NotRequired[int]  # 仅 Anthropic 拆分；cacheWrite 的子集
    reasoning: NotRequired[int]     # 推理 token（output 的子集）


class AssistantMessage(TypedDict):
    """
    Assistant 回复消息
    """

    role: Literal["assistant"]
    content: list[ContentBlock]             # 输出内容
    api: str                                # 使用哪个 API
    provider: str                           # Provider 名称
    model: str                              # 模型名称
    usage: Usage                            # Token 使用统计（必填）
    stopReason: StopReason                  # 停止原因（必填）
    timestamp: int                          # 时间戳（必填）

    # 可选字段
    responseModel: NotRequired[str]        # 实际响应的模型（如 OpenRouter auto）
    responseId: NotRequired[str]           # 上游 response id
    diagnostics: NotRequired[list[Any]]    # AssistantMessageDiagnostic[]
    errorMessage: NotRequired[str | None]  # 错误信息
    rawStopReason: NotRequired[str]        # 上游原始 finish_reason


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
    isError: bool                     # 是否执行失败（必填，对齐 TS）
    timestamp: int                    # Unix 毫秒时间戳（必填，对齐 TS）
    addedToolNames: NotRequired[list[str] | None]

    # 可选字段
    details: NotRequired[Any]   # 工具执行的额外细节
    usage: NotRequired[Usage]   # 工具自身 usage（不计入主 LLM 上下文）


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

    # 受约束采样配置（json_schema / grammar）；False 或 None 表示关闭
    constrainedSampling: Literal[False] | ConstrainedSamplingConfig | None = None


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


# =========================================================
# 模型成本（对齐 TS ModelCost）
# =========================================================

@dataclass(slots=True)
class ModelCostRates:
    """$ / 百万 token"""

    input: float = 0.0
    output: float = 0.0
    cacheRead: float = 0.0
    cacheWrite: float = 0.0

@dataclass(slots=True)
class ModelCostTier(ModelCostRates):
    """输入用量超过该 token 数时启用此档价格"""

    inputTokensAbove: int = 0

@dataclass(slots=True)
class ModelCost(ModelCostRates):
    """请求级价格档位；最高匹配档位适用于整个请求"""

    tiers: list[ModelCostTier] = field(default_factory=list)


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
    cost: ModelCost = field(default_factory=ModelCost)  # Token 成本（对齐 TS ModelCost）
    maxTokens: int = 4096               # 最大 Token
    reasoning: bool = False             # 是否支持 Thinking（对齐 TS reasoning）
    supportsToolCalling: bool = True    # 是否支持 Tool Calling
    supportsImages: bool = False        # 是否支持图片

    # ---- 对齐 TS 新增字段（阶段 1：带默认值，不破坏现有构造）----
    baseUrl: str = ""                                   # 模型级 base url（可选覆盖 Provider 级）
    contextWindow: int = 0                              # 上下文窗口 token 数
    headers: dict[str, str] | None = None               # 模型级自定义头
    compat: Any = None                                  # 四个 *Compat TypedDict 之一（Any + 文档近似）
    thinkingLevelMap: ThinkingLevelMap | None = None    # pi 思考级别 -> provider 值映射

    def capabilities(self) -> list[str]:
        """返回模型支持的能力标签列表（thinking / tools / images）。

        供 CLI、日志等展示层消费，避免各调用方重复拼装能力逻辑。
        """
        caps: list[str] = []
        if self.reasoning:
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

    # System Prompt（对齐 TS Context.systemPrompt）
    systemPrompt: str | None = None


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
    #
    # 有意分歧：TS 为 AbortSignal，Python 用 asyncio.Event。
    signal: NotRequired[asyncio.Event]

    # ---- 对齐 TS 新增字段（阶段 1）----

    # 自定义 fetch 实现；Python 无 fetch，保留用于对齐（实现时可映射 httpx 或忽略）
    fetch: NotRequired[FetchFunction]

    # 首选传输协议（支持多传输的 provider 使用）
    transport: NotRequired[Transport]

    # 提示缓存保留偏好。默认 "short"
    cacheRetention: NotRequired[CacheRetention]

    # 会话标识（支持会话缓存的 provider 使用）
    sessionId: NotRequired[str]

    # 发送前检查/替换 payload 的回调；返回 None 表示保持原样
    onPayload: NotRequired[Callable[..., Any]]

    # HTTP 响应接收后、消费 body 前的回调
    onResponse: NotRequired[Callable[..., Any]]

    # HTTP 请求超时（毫秒）
    timeoutMs: NotRequired[int]

    # WebSocket 连接超时（毫秒）；仅覆盖握手阶段
    websocketConnectTimeoutMs: NotRequired[int]

    # 附加元数据（provider 提取其认识的字段，忽略其余）
    metadata: NotRequired[dict[str, Any]]

    # Provider 作用域环境变量（优先于 os.environ）
    env: NotRequired[ProviderEnv]


# =========================================================
# ProviderStreamOptions / SimpleStreamOptions
# =========================================================

# TS: StreamOptions & Record<string, unknown>
ProviderStreamOptions = StreamOptions


class SimpleStreamOptions(StreamOptions):
    """统一选项 + 推理参数（streamSimple / completeSimple 使用）"""

    # 推理级别
    reasoning: NotRequired[ThinkingLevel]

    # 各推理级别的自定义 token 预算（仅 token-based provider）
    thinkingBudgets: NotRequired[ThinkingBudgets]


# =========================================================
# Stream Event（流事件）
# =========================================================
#
# EventStream 会不断产生这些事件。
#
# start
# text_delta
# thinking_delta
# toolcall_delta
# done
#
# =========================================================


class StartEvent(TypedDict):
    """流开始（首个事件；partial 为当前空累积状态）"""

    type: Literal["start"]
    partial: AssistantMessage


class TextStartEvent(TypedDict):
    """文本块开始"""

    type: Literal["text_start"]
    contentIndex: int
    partial: AssistantMessage


class TextDeltaEvent(TypedDict):
    """文本增量"""

    type: Literal["text_delta"]
    contentIndex: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(TypedDict):
    """文本块结束"""

    type: Literal["text_end"]
    contentIndex: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(TypedDict):
    """思考块开始"""

    type: Literal["thinking_start"]
    contentIndex: int
    partial: AssistantMessage


class ThinkingDeltaEvent(TypedDict):
    """思考内容增量"""

    type: Literal["thinking_delta"]
    contentIndex: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(TypedDict):
    """思考块结束"""

    type: Literal["thinking_end"]
    contentIndex: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(TypedDict):
    """工具调用块开始"""

    type: Literal["toolcall_start"]
    contentIndex: int
    partial: AssistantMessage


class ToolCallDeltaEvent(TypedDict):
    """工具调用增量（原始 arguments JSON 字符串片段）"""

    type: Literal["toolcall_delta"]
    contentIndex: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(TypedDict):
    """工具调用结束（携带已解析的 ToolCall）"""

    type: Literal["toolcall_end"]
    contentIndex: int
    toolCall: ToolCall
    partial: AssistantMessage


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
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]

# =========================================================
# 图片生成类型（对齐 TS；类型先行，无运行时实现）
# =========================================================

KnownImagesApi = Literal["openrouter-images"]
KnownImagesProvider = Literal["openrouter"]

ImagesApi = KnownImagesApi | str
ImagesProviderId = KnownImagesProvider | str

ImagesStopReason = Literal["stop", "error", "aborted"]

ImagesInputContent = TextContent | ImageContent
ImagesOutputContent = TextContent | ImageContent


class ImagesContext(TypedDict):
    input: list[ImagesInputContent]


class AssistantImages(TypedDict):
    api: ImagesApi
    provider: ImagesProviderId
    model: str
    output: list[ImagesOutputContent]
    responseId: NotRequired[str]
    usage: NotRequired[Usage]
    stopReason: ImagesStopReason
    errorMessage: NotRequired[str]
    timestamp: int


@dataclass(slots=True)
class ImagesModel:
    id: str
    api: ImagesApi
    provider: ImagesProviderId
    name: str = ""
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    headers: dict[str, str] | None = None


class ImagesOptions(TypedDict, total=False):
    """图片生成请求参数（对齐 TS ImagesOptions）"""

    signal: NotRequired[asyncio.Event]
    apiKey: str
    fetch: FetchFunction
    env: ProviderEnv
    onPayload: Callable[..., Any]
    onResponse: Callable[..., Any]
    headers: ProviderHeaders
    timeoutMs: int
    maxRetries: int
    maxRetryDelayMs: int
    metadata: dict[str, Any]


ProviderImagesOptions = ImagesOptions


# =========================================================
# 兼容配置（Compat）与路由（对齐 TS；Python 用文档化近似）
# =========================================================

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
        "openai", "openrouter", "deepseek", "together", "zai", "qwen",
        "chat-template", "qwen-chat-template", "string-thinking", "ant-ling",
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


# =========================================================
# 流契约（对齐 TS ProviderStreams / ProviderImages）
# =========================================================

class ProviderStreams(Protocol):
    """API 实现模块的统一流契约（stream + streamSimple）"""

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> "AssistantMessageEventStream": ...

    def streamSimple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> "AssistantMessageEventStream": ...


class ProviderImages(Protocol):
    """图片 API 实现模块的统一契约（generateImages）"""

    def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> "Awaitable[AssistantImages]": ...


# 流函数类型别名（从 provider.py 上移）。
#
# 返回值用字符串前向引用：避免 _types ↔ _event_stream 循环导入。
StreamFunction = Callable[
    [Model, Context, StreamOptions | None],
    "Awaitable[AssistantMessageEventStream]",
]

ImagesFunction = Callable[
    [ImagesModel, ImagesContext, ImagesOptions | None],
    "Awaitable[AssistantImages]",
]


def now_ms() -> int:
    """当前 Unix 时间戳（毫秒）。

    用于构造 Message / AssistantImages 的 timestamp 字段。
    """
    return int(time.time() * 1000)


# 暴露接口
__all__ = [
    "TextContent",
    "ImageContent",
    "ToolCall",
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
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    "AssistantMessageEvent",

    # 阶段 2 新增
    "Cost",
    "now_ms",

    # 阶段 1 新增
    "TextSignatureV1",
    "ThinkingLevelMap",
    "ThinkingBudgets",
    "ChatTemplateKwargVar",
    "ChatTemplateKwargValue",
    "ProviderEnv",
    "ProviderHeaders",
    "FetchFunction",
    "SessionAffinityFormat",
    "ProviderResponse",
    "GrammarFormat",
    "GrammarVariants",
    "JsonSchemaSampling",
    "GrammarSampling",
    "ConstrainedSamplingConfig",
    "ModelCostRates",
    "ModelCostTier",
    "ModelCost",
    "ProviderStreamOptions",
    "SimpleStreamOptions",
    "ImagesOptions",
    "ProviderImagesOptions",
    "KnownImagesApi",
    "KnownImagesProvider",
    "ImagesApi",
    "ImagesProviderId",
    "ImagesStopReason",
    "ImagesInputContent",
    "ImagesOutputContent",
    "ImagesContext",
    "AssistantImages",
    "ImagesModel",
    "OpenAICompletionsCompat",
    "OpenAIResponsesCompat",
    "AnthropicMessagesCompat",
    "BedrockCompat",
    "OpenRouterRouting",
    "VercelGatewayRouting",
    "ProviderStreams",
    "ProviderImages",
    "StreamFunction",
    "ImagesFunction",
]
