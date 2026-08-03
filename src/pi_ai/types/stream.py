"""pi_ai.types.stream — 流事件（Event）与流式请求参数（StreamOptions）。

事件协议（12 种）：

    start / text_start / text_delta / text_end /
    thinking_start / thinking_delta / thinking_end /
    toolcall_start / toolcall_delta / toolcall_end /
    done / error

所有事件继承 BaseEvent（type 判别字段 + 可选 timestamp），
增量事件（start / *_start / *_delta / *_end）都携带 partial 快照。
"""

import asyncio

from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
)

from .common import (
    AsyncHTTPClient,
    CacheRetention,
    ProviderEnv,
    ProviderHeaders,
    ThinkingBudgets,
    ThinkingLevel,
    Transport,
)
from .content import ToolCall
from .context import Context
from .message import AssistantMessage
from .model import Model


# =========================================================
# Stream Event（流事件）
# =========================================================


class BaseEvent(TypedDict):
    """事件基础协议：所有流事件共享 type 判别字段。"""

    type: str
    timestamp: NotRequired[int]  # 可选：事件产生时的 Unix 毫秒时间戳


class StartEvent(BaseEvent):
    """流开始（首个事件；partial 为当前空累积状态）"""

    type: Literal["start"]
    partial: AssistantMessage


class TextStartEvent(BaseEvent):
    """文本块开始"""

    type: Literal["text_start"]
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(BaseEvent):
    """文本增量"""

    type: Literal["text_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(BaseEvent):
    """文本块结束"""

    type: Literal["text_end"]
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(BaseEvent):
    """思考块开始"""

    type: Literal["thinking_start"]
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(BaseEvent):
    """思考内容增量"""

    type: Literal["thinking_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(BaseEvent):
    """思考块结束"""

    type: Literal["thinking_end"]
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(BaseEvent):
    """工具调用块开始"""

    type: Literal["toolcall_start"]
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(BaseEvent):
    """工具调用增量（原始 arguments JSON 字符串片段）"""

    type: Literal["toolcall_delta"]
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(BaseEvent):
    """工具调用结束（携带已解析的 ToolCall）"""

    type: Literal["toolcall_end"]
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(BaseEvent):
    """流结束"""

    type: Literal["done"]
    # 成功终止原因
    reason: Literal["stop", "length", "tool_call"]
    message: AssistantMessage


class ErrorEvent(BaseEvent):
    """流异常结束"""

    type: Literal["error"]
    # 异常终止原因
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


# =========================================================
# StreamOptions
# =========================================================


class StreamOptions(TypedDict, total=False):
    """
    流式请求参数。
    """

    temperature: float
    max_tokens: int
    api_key: str
    # Provider 层解析后的 Base URL（注册表分发时注入；None 时回退 model.base_url）
    base_url: NotRequired[str]
    thinking_budget: int | None
    thinking_enabled: bool | None
    headers: dict[str, str | None]

    # 最大重试次数（provider 层客户端重试上限）。
    max_retries: int

    # 最大重试延迟（毫秒）。
    #
    # 当服务器要求等待超过该值时直接失败，
    # 交由上层重试逻辑处理。
    max_retry_delay_ms: int

    # 可选的中止信号（asyncio.Event）。
    #
    # 支持流式中止的 Provider（例如 Faux Provider）
    # 会在流式输出过程中检查它，
    # 一旦被 set 即以 aborted 结束。
    signal: NotRequired[asyncio.Event]

    # 可注入的异步 HTTP 客户端（如 httpx.AsyncClient）；None 用默认。
    http_client: NotRequired[AsyncHTTPClient]

    # 首选传输协议（支持多传输的 provider 使用）
    transport: NotRequired[Transport]

    # 提示缓存保留偏好。默认 "short"
    cache_retention: NotRequired[CacheRetention]

    # 会话标识（支持会话缓存的 provider 使用）
    session_id: NotRequired[str]

    # 发送前检查/替换 payload 的回调；返回 None 表示保持原样
    on_payload: NotRequired[Callable[..., Any]]

    # HTTP 响应接收后、消费 body 前的回调
    on_response: NotRequired[Callable[..., Any]]

    # HTTP 请求超时（毫秒）
    timeout_ms: NotRequired[int]

    # WebSocket 连接超时（毫秒）；仅覆盖握手阶段
    websocket_connect_timeout_ms: NotRequired[int]

    # 附加元数据（provider 提取其认识的字段，忽略其余）
    metadata: NotRequired[dict[str, Any]]

    # Provider 作用域环境变量（优先于 os.environ）
    env: NotRequired[ProviderEnv]


ProviderStreamOptions = StreamOptions


class SimpleStreamOptions(StreamOptions):
    """统一选项 + 推理参数（streamSimple / completeSimple 使用）"""

    # 推理级别
    reasoning: NotRequired[ThinkingLevel]

    # 各推理级别的自定义 token 预算（仅 token-based provider）
    thinking_budgets: NotRequired[ThinkingBudgets]


# =========================================================
# 流契约（ProviderStreams / ProviderImages / 函数别名）
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


# 流函数类型别名（返回值用字符串前向引用：避免与 utils._event_stream 循环导入）
StreamFunction = Callable[
    [Model, Context, StreamOptions | None],
    "Awaitable[AssistantMessageEventStream]",
]


from .image import ImagesContext, ImagesModel, ImagesOptions  # noqa: E402


class ProviderImages(Protocol):
    """图片 API 实现模块的统一契约（generateImages）"""

    def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> "Awaitable[AssistantImages]": ...


ImagesFunction = Callable[
    [ImagesModel, ImagesContext, ImagesOptions | None],
    "Awaitable[AssistantImages]",
]


from .image import AssistantImages  # noqa: E402


__all__ = [
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
]
