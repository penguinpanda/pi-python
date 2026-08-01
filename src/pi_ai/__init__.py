"""
pi-ai  统一管理不同模型的 API,返回一个一致的 AI 调用接口。

暂只支持 Openai 与 Deepseek 
"""

from ._event_stream import AssistantMessageEventStream, EventStream
from ._types import (
    AssistantMessage,
    AssistantMessageEvent,
    ContentBlock,
    Context,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    StreamOptions,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ThinkingDeltaEvent,
    Tool,
    ToolCallContent,
    ToolCallDeltaEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
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
    OPENAI_MODELS,
    deepseek_provider,
    openai_provider,
)


def create_default_models() -> Models:
    """ 创建一个预加载了OpenAI和DeepSeek的Models实例。 """
    models = Models()
    models.add_provider(openai_provider())
    models.add_provider(deepseek_provider())
    return models

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

    # Model lists
    "OPENAI_MODELS",
    "DEEPSEEK_MODELS",

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
    "ToolCallContent",
    "ThinkingContent",
    "Usage",
    "StreamOptions",

    # Events
    "AssistantMessageEvent",
    "DeltaEvent",
    "ToolCallDeltaEvent",
    "ThinkingDeltaEvent",
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
