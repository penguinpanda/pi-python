"""
pi-agent-core 类型定义（Core Types）

本模块定义 Agent 层的所有公共类型。

分层策略（与 pi_ai 一致）：

1. TypedDict
--------------
用于"运行时数据"（携带数据的事件、配置片段）:
- AgentEvent（10 种判别联合类型）
- BeforeToolCallResult / AfterToolCallResult / AgentLoopTurnUpdate

2. dataclass
--------------
用于"Python 内部对象":
- AgentTool / AgentToolResult / AgentContext / AgentState / AgentLoopConfig

3. 类型别名 / Protocol
--------------
用于依赖注入抽象:
- StreamFn / StreamOptions / AgentEventSink
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# 从 pi_ai 直接复用的类型
# ---------------------------------------------------------------------------
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    CacheRetention,
    ImageContent,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils._event_stream import AssistantMessageEventStream

# Context 类型（pi_ai 中是 dataclass，可通过 pi_ai 导入）
from pi_ai import Context as LlmContext
from pi_ai import RetryPolicy

# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------
# Agent 层的"消息"即 pi_ai.Message 联合类型（含系统/用户/助手/工具结果，
# 以及 pi_ai 的 AgentMessage 通用扩展 role）。
# TypeScript 版通过 Declaration Merging 扩展 CustomAgentMessages，
# Python 不支持声明合并，用 pi_ai.types.AgentMessage 承载通用 Agent role。
AgentMessage = Message

# ---------------------------------------------------------------------------
# StreamFn（依赖注入抽象）
# ---------------------------------------------------------------------------
# LLM 调用的统一签名。
# 不抛异常 —— 错误通过事件流中的 stop_reason="error" 编码。
# 通过依赖注入解耦具体 provider。
StreamFn = Callable[
    [Model, LlmContext, StreamOptions | None],
    Awaitable[AssistantMessageEventStream],
]

# ---------------------------------------------------------------------------
# AgentTool
# ---------------------------------------------------------------------------

# 工具执行模式（对齐 TS ToolExecutionMode）：
# - "sequential": 每个工具调用先准备、再执行、再完成，然后才轮到下一个
# - "parallel": 所有工具顺序准备，允许的工具并发执行；tool_execution_end 按完成
#   顺序发出，ToolResultMessage 消息随后按 assistant 原始顺序发出
ToolExecutionMode = Literal["sequential", "parallel"]

# 消息队列消费策略（对齐 TS QueueMode）：
# - "all": 一次 drain 全部注入
# - "one-at-a-time": 每次 drain 只消费最早一条，剩余留待后续 drain 点
QueueMode = Literal["all", "one-at-a-time"]


@dataclass(slots=True)
class AgentToolResult:
    """工具执行结果。

    content 中的 TextContent/ImageContent 会作为 toolResult 消息传给 LLM。
    """
    content: list[TextContent | ImageContent]
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool = False


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass(slots=True)
class AgentTool:
    """Agent 工具定义。

    input_schema 为 JSON Schema dict，用于 LLM 的工具选择。
    execute 回调接收 (tool_call_id, params, signal?, on_update?) → AgentToolResult。

    生命周期钩子（可选）：
    - before_execute(args, context) → dict | None：执行前调用，返回 dict 替换参数
    - after_execute(result) → AgentToolResult | None：执行后调用，返回新值替换结果
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    label: str
    execute: Callable[..., Awaitable[AgentToolResult]]
    # 单工具执行模式（对齐 TS executionMode，省略时按默认并行）。
    # 批次内任一工具声明 "sequential" → 整批回退为顺序执行。
    execution_mode: ToolExecutionMode = "parallel"

    # ---- 生命周期钩子（可选，默认 None 不改变现有行为）----

    # 执行前钩子：收到参数 dict 与执行上下文。
    # 返回 None 表示放行；返回 dict 可替换传给 execute 的参数。
    before_execute: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None

    # 执行后钩子：收到 execute 的返回值。
    # 返回 None 保持原结果；返回新值则替换最终结果。
    after_execute: Callable[[Any], Awaitable[Any]] | None = None


# ---------------------------------------------------------------------------
# AgentState / AgentContext
# ---------------------------------------------------------------------------

ThinkingLevel = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max"
]


@dataclass
class AgentState:
    """可观察的 Agent 运行时只读状态。

    tools 和 messages 赋值时自动防御性复制。
    """
    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel = "off"
    _tools: list[AgentTool] = field(default_factory=list, repr=False)
    _messages: list[AgentMessage] = field(default_factory=list, repr=False)
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None

    # -- property: tools（防御性复制）--
    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    # -- property: messages（防御性复制）--
    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)

    # -- 内部使用：直接访问底层 list 用于累积消息 --
    def _append_message(self, msg: AgentMessage) -> None:
        self._messages.append(msg)

    def _append_tool(self, tool: AgentTool) -> None:
        self._tools.append(tool)


@dataclass(slots=True)
class AgentContext:
    """传入 agent loop 的不可变上下文快照。"""
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


# ---------------------------------------------------------------------------
# AgentEvent（10 种判别联合事件）
# ---------------------------------------------------------------------------

class AgentStartEvent(TypedDict):
    type: Literal["agent_start"]


class AgentEndEvent(TypedDict):
    type: Literal["agent_end"]
    messages: list[AgentMessage]


class TurnStartEvent(TypedDict):
    type: Literal["turn_start"]


class TurnEndEvent(TypedDict):
    type: Literal["turn_end"]
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(TypedDict):
    type: Literal["message_start"]
    message: AgentMessage


class MessageUpdateEvent(TypedDict):
    type: Literal["message_update"]
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class MessageEndEvent(TypedDict):
    type: Literal["message_end"]
    message: AgentMessage


class ToolExecutionStartEvent(TypedDict):
    type: Literal["tool_execution_start"]
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(TypedDict):
    type: Literal["tool_execution_update"]
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class ToolExecutionEndEvent(TypedDict):
    type: Literal["tool_execution_end"]
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


class AutoRetryStartEvent(TypedDict):
    """重试已计划：退避等待开始前发射（对齐 TS auto_retry_start）。"""
    type: Literal["auto_retry_start"]
    attempt: int          # 本次重试序号（从 1 起）
    max_attempts: int     # 策略重试上限
    delay_ms: float       # 退避延迟（毫秒）
    error_message: str    # 触发重试的错误消息


class AutoRetryEndEvent(TypedDict):
    """重试循环结束（成功 / 放弃）发射（对齐 TS auto_retry_end）。"""
    type: Literal["auto_retry_end"]
    success: bool          # 是否最终成功
    attempt: int           # 结束时的重试序号
    final_error: str | None  # 最终错误（成功时为 None）


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | AutoRetryStartEvent
    | AutoRetryEndEvent
)

# 事件发射回调
AgentEventSink = Callable[[AgentEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# AgentLoopConfig（可注入钩子）
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BeforeToolCallResult:
    """beforeToolCall 返回值，block=True 阻止工具执行。"""
    block: bool = False
    reason: str = ""


@dataclass(slots=True)
class AfterToolCallResult:
    """afterToolCall 返回值，字段级覆盖工具结果。"""
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    """beforeToolCall 专用 context（对齐 TS BeforeToolCallContext）。"""
    assistant_message: AssistantMessage  # 触发工具调用的 assistant 消息
    tool_call: ToolCall                  # 原始 toolCall 块
    args: Any                            # 校验后的参数
    context: AgentContext                # 当前 Agent 上下文（含本轮 messages）


@dataclass(slots=True)
class AfterToolCallContext:
    """afterToolCall 专用 context（对齐 TS AfterToolCallContext）。"""
    assistant_message: AssistantMessage  # 触发工具调用的 assistant 消息
    tool_call: ToolCall                  # 原始 toolCall 块
    args: Any                            # 校验后的参数
    result: AgentToolResult              # afterToolCall 覆盖前的执行结果
    is_error: bool                       # 当前是否按错误处理
    context: AgentContext                # 当前 Agent 上下文


@dataclass(slots=True)
class AgentLoopTurnUpdate:
    """prepareNextTurn 返回值，替换下一轮状态。"""
    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True)
class AgentLoopConfig:
    """Agent 循环的所有可注入配置。

    必需字段:
        model: 模型选择
        convert_to_llm: AgentMessage → LLM Message 转换（唯一转换点）

    可选钩子（均为 Callable | None）:
        transform_context: 上下文预处理
        get_api_key: 动态认证密钥解析
        get_steering_messages: 轮间注入引导消息
        get_follow_up_messages: Agent 即将停止时注入后续消息
        should_stop_after_turn: 提前终止判断
        prepare_next_turn: 轮间状态准备
        before_tool_call: 工具执行前拦截
        after_tool_call: 工具执行后拦截

    配置:
        tool_execution: 工具执行模式（最小核心仅 "sequential"）
    """
    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]]

    # 可选钩子
    transform_context: (
        Callable[[list[AgentMessage]], Awaitable[list[AgentMessage]]] | None
    ) = None
    get_api_key: Callable[[str], str | None] | None = None
    should_stop_after_turn: (
        Callable[[AgentContext], bool | Awaitable[bool]] | None
    ) = None
    prepare_next_turn: (
        Callable[
            [AgentContext],
            AgentLoopTurnUpdate | Awaitable[AgentLoopTurnUpdate | None] | None,
        ]
        | None
    ) = None
    before_tool_call: (
        Callable[
            [BeforeToolCallContext],
            BeforeToolCallResult | Awaitable[BeforeToolCallResult | None] | None,
        ]
        | None
    ) = None
    after_tool_call: (
        Callable[
            [AfterToolCallContext],
            AfterToolCallResult
            | Awaitable[AfterToolCallResult | None]
            | None,
        ]
        | None
    ) = None

    # 消息队列轮询钩子（双重嵌套循环 1.1 的核心输入）。
    # 每次调用返回待注入的 AgentMessage 列表；无消息时返回 []。
    get_steering_messages: (
        Callable[[], Awaitable[list[AgentMessage]]] | None
    ) = None
    get_follow_up_messages: (
        Callable[[], Awaitable[list[AgentMessage]]] | None
    ) = None

    # 配置
    # 默认并行（对齐 TS Agent.toolExecution 默认值）。
    tool_execution: ToolExecutionMode = "parallel"

    # 提示缓存与会话标识（透传给 StreamOptions）
    session_id: str | None = None
    cache_retention: CacheRetention | None = None

    # 重试策略。None 表示使用默认策略（enabled=True, max_retries=3）。
    # 显式传入 RetryPolicy(enabled=False) 可关闭重试。
    retry_policy: RetryPolicy | None = None
