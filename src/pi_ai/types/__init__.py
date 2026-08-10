"""pi_ai.types — 类型定义包。

拆分自原 `pi_ai/_types.py`（该文件保留为兼容 re-export 层）。

模块划分：

    common.py    跨领域基础类型（枚举 / 协议 / 工具函数）
    content.py   消息内容块（ContentBlock）
    message.py   消息（Message）
    tool.py      工具定义（Tool）
    model.py     模型元数据（Model / ModelCost）
    context.py   请求上下文（Context / MemoryStore）
    stream.py    流事件与流式参数（Event / StreamOptions）
    image.py     图片生成类型
    compat.py    Provider 兼容配置（Compat）
    trace.py     可观测性（Trace / TraceSpan）

本模块汇总导出所有公共类型。
"""

from .common import (
    AsyncHTTPClient,
    CacheRetention,
    ChatTemplateKwargValue,
    ChatTemplateKwargVar,
    ConstrainedSamplingConfig,
    GrammarFormat,
    GrammarSampling,
    GrammarVariants,
    JsonSchemaSampling,
    ModelThinkingLevel,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SessionAffinityFormat,
    StopReason,
    ThinkingBudgets,
    ThinkingLevel,
    ThinkingLevelMap,
    Transport,
    now_ms,
)
from .content import (
    BaseContent,
    CodeContent,
    ContentBlock,
    ImageContent,
    TextContent,
    TextSignatureV1,
    ThinkingContent,
    ToolCall,
)
from .message import (
    AgentMessage,
    AssistantMessage,
    AssistantMessageDiagnostic,
    BaseMessage,
    Cost,
    DeferredHandle,
    Message,
    SystemMessage,
    ToolDetails,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .tool import Tool
from .compat import (
    AnthropicMessagesCompat,
    BedrockCompat,
    ModelCompat,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    OpenRouterRouting,
    VercelGatewayRouting,
)
from .model import (
    ApiId,
    Model,
    ModelCost,
    ModelCostRates,
    ModelCostTier,
    ModelInput,
    ModelOutput,
    ProviderId,
)
from .context import Context, MemoryStore
from .image import (
    AssistantImages,
    ImagesApi,
    ImagesContext,
    ImagesInputContent,
    ImagesModel,
    ImagesOptions,
    ImagesOutputContent,
    ImagesProviderId,
    ImagesStopReason,
    KnownImagesApi,
    KnownImagesProvider,
    ProviderImagesOptions,
)
from .stream import (
    AssistantMessageEvent,
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    ImagesFunction,
    ProviderImages,
    ProviderStreamOptions,
    ProviderStreams,
    SimpleStreamOptions,
    StartEvent,
    StreamFunction,
    StreamOptions,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from .trace import Trace, TraceSpan


__all__ = [
    # common
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
    # content
    "BaseContent",
    "TextContent",
    "ImageContent",
    "ToolCall",
    "ThinkingContent",
    "CodeContent",
    "ContentBlock",
    "TextSignatureV1",
    # message
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "AgentMessage",
    "Message",
    "Cost",
    "Usage",
    "AssistantMessageDiagnostic",
    "DeferredHandle",
    "ToolDetails",
    # tool
    "Tool",
    # model
    "ApiId",
    "ProviderId",
    "ModelInput",
    "ModelOutput",
    "ModelCostRates",
    "ModelCostTier",
    "ModelCost",
    "Model",
    # context
    "Context",
    "MemoryStore",
    # stream
    "BaseEvent",
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
    "StreamOptions",
    "ProviderStreamOptions",
    "SimpleStreamOptions",
    "ProviderStreams",
    "ProviderImages",
    "StreamFunction",
    "ImagesFunction",
    # image
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
    # compat
    "OpenAICompletionsCompat",
    "OpenAIResponsesCompat",
    "AnthropicMessagesCompat",
    "BedrockCompat",
    "OpenRouterRouting",
    "VercelGatewayRouting",
    "ModelCompat",
    # trace
    "Trace",
    "TraceSpan",
]
