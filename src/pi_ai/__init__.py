"""
pi-ai  统一管理不同模型的 API,返回一个一致的 AI 调用接口。

暂支持 Openai、Deepseek、Ollama 与 Faux(测试用假 Provider)
"""

from ._event_stream import AssistantMessageEventStream, EventStream
from ._types import (
    AnthropicMessagesCompat,
    AssistantImages,
    AssistantMessage,
    AssistantMessageEvent,
    BedrockCompat,
    CacheRetention,
    ChatTemplateKwargValue,
    ChatTemplateKwargVar,
    ConstrainedSamplingConfig,
    ContentBlock,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    FetchFunction,
    GrammarFormat,
    GrammarSampling,
    GrammarVariants,
    ImageContent,
    ImagesApi,
    ImagesContext,
    ImagesFunction,
    ImagesInputContent,
    ImagesModel,
    ImagesOptions,
    ImagesOutputContent,
    ImagesProviderId,
    ImagesStopReason,
    JsonSchemaSampling,
    KnownImagesApi,
    KnownImagesProvider,
    Message,
    Model,
    ModelCost,
    ModelCostRates,
    ModelCostTier,
    ModelThinkingLevel,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    OpenRouterRouting,
    ProviderEnv,
    ProviderHeaders,
    ProviderImages,
    ProviderImagesOptions,
    ProviderResponse,
    ProviderStreamOptions,
    ProviderStreams,
    SessionAffinityFormat,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamFunction,
    StreamOptions,
    SystemMessage,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextSignatureV1,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingLevelMap,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Transport,
    Usage,
    UserMessage,
    VercelGatewayRouting,
    now_ms,
)
from .auth import (
    ApiKeyCredential,
    InMemoryCredentialStore,
    env_api_key_auth,
)
from .models import Models
from .provider import Provider, create_provider
from .providers import (
    DEEPSEEK_MODELS,
    FAUX_MODEL,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    deepseek_provider,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
    ollama_provider,
    openai_provider,
)
from .providers.all import create_default_models


# def create_default_models() -> Models:
#     """ 创建一个预加载了OpenAI和DeepSeek的Models实例。 """
#     models = Models()
#     models.add_provider(openai_provider())
#     models.add_provider(deepseek_provider())
#     return models

__all__ = [
    # Core registry
    "Models",
    "create_default_models",

    # Provider
    "Provider",
    "create_provider",
    
    # Provider factories
    "openai_provider",
    "deepseek_provider",
    "ollama_provider",
    "faux_provider",

    # Model lists
    "OPENAI_MODELS",
    "DEEPSEEK_MODELS",
    "OLLAMA_MODELS",
    "FAUX_MODEL",

    # Faux helpers
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",

    # Types
    "Model",
    "Context",
    "Tool",
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "ContentBlock",
    "TextContent",
    "ImageContent",
    "ToolCall",
    "ThinkingContent",
    "Usage",
    "Cost",
    "now_ms",
    "StreamOptions",

    # Types (阶段 1 新增)
    "TextSignatureV1",
    "ThinkingLevel",
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

    # Events
    "AssistantMessageEvent",
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

    # Stream
    "AssistantMessageEventStream",
    "EventStream",
    
    # Auth
    "env_api_key_auth",
    "ApiKeyCredential",
    "InMemoryCredentialStore",
]
