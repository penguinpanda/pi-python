"""pi_ai.types.message — 消息（Message）。

对话历史全部由 Message 组成：

    Context.messages

    [
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        AgentMessage,      # Agent 运行时扩展（可选的通用 Agent role）
    ]

扩展机制：所有消息继承 BaseMessage（role + timestamp），
Agent 层可通过 AgentMessage 携带任意 Agent role（planner/observation/memory...）。
"""

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

from .common import StopReason
from .content import ContentBlock, ImageContent, TextContent


class BaseMessage(TypedDict):
    """消息基础协议（标记基类）。

    判别字段 ``role`` 由各具体子类声明为唯一 Literal，
    以构成 Message 可辨识联合（mypy 不支持基类收窄字段类型）。
    """

    pass


class SystemMessage(BaseMessage):
    """System Prompt"""

    role: Literal["system"]
    content: str
    timestamp: NotRequired[int]  # Unix 毫秒时间戳（可选）


class UserMessage(BaseMessage):
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
    content: str | list[TextContent | ImageContent]
    timestamp: int  # Unix 毫秒时间戳（必填）


class Cost(TypedDict):
    """成本统计"""

    input: float
    output: float
    cache_read: float
    cache_write: float
    total: float


class Usage(TypedDict):
    """
    Token 使用统计
    """

    input: int
    output: int
    cache_read: int
    cache_write: int
    total_tokens: int
    cost: Cost

    # 可选字段
    cache_write_1h: NotRequired[int]  # 仅 Anthropic 拆分；cache_write 的子集
    reasoning: NotRequired[int]  # 推理 token（output 的子集）


class DiagnosticErrorInfo(TypedDict, total=False):
    """诊断错误信息（`DiagnosticErrorInfo`）。"""

    name: str  # 错误类型名（如 ProviderError）
    message: str  # 错误消息
    stack: str  # 堆栈（可用时）
    code: str | int  # 错误码（如 HTTP 429）


class AssistantMessageDiagnostic(TypedDict, total=False):
    """助手消息诊断条目（`AssistantMessageDiagnostic`）。

    记录 provider / runtime 失败与恢复信息，供调试定位。
    """

    type: str  # 诊断类型（如 provider_error / retry_exhausted）
    timestamp: int  # 诊断时间戳（Unix 毫秒）
    error: DiagnosticErrorInfo  # 规范化错误信息
    details: dict[str, Any]  # 附加信息


class AssistantMessage(BaseMessage):
    """
    Assistant 回复消息
    """

    role: Literal["assistant"]
    content: list[ContentBlock]  # 输出内容
    api: str  # 使用哪个 API
    provider: str  # Provider 名称
    model: str  # 模型名称
    usage: NotRequired[Usage]  # Token 使用统计（部分 provider 不提供）
    stop_reason: NotRequired[StopReason]  # 停止原因
    timestamp: int  # 时间戳（必填）

    # 可选字段
    response_model: NotRequired[str]  # 实际响应的模型（如 OpenRouter auto）
    response_id: NotRequired[str]  # 上游 response id
    diagnostics: NotRequired[list[AssistantMessageDiagnostic]]  # 诊断信息
    error_message: NotRequired[str | None]  # 错误信息
    raw_stop_reason: NotRequired[str]  # 上游原始 finish_reason
    responses_items: NotRequired[list[dict[str, Any]]]  # Responses 原始 item（如 web_search_call）


class ToolDetails(TypedDict, total=False):
    """工具执行附加详情的类型化选项（details 字段推荐结构）。"""

    duration_ms: int
    error: str | None
    input: dict[str, Any]
    output: dict[str, Any]


class ToolResultMessage(BaseMessage):
    """
    Tool 返回结果

    工具执行完成以后，
    会将结果作为新的 Message
    再发给模型。
    """

    role: Literal["toolResult"]
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool  # 是否执行失败
    timestamp: int  # Unix 毫秒时间戳
    added_tool_names: NotRequired[list[str] | None]

    # 可选字段
    details: NotRequired[Any]  # 工具执行的额外细节（推荐用 ToolDetails 结构）
    usage: NotRequired[Usage]  # 工具自身 usage（不计入主 LLM 上下文）


class AgentMessage(BaseMessage):
    """Agent 运行时消息（通用扩展 role）。

    用于多 Agent 协作 / ReAct / 规划器等场景，
    携带 Chat API 原生不认识的 role：

        planner / observation / memory / function /
        developer / critic / reflection / ...

    注意：role 刻意排除 "system" / "user" / "assistant" / "toolResult"
    （这四个已由专用 Message 类型占用，避免判别联合冲突）。
    """

    role: Literal[
        "planner",
        "observation",
        "memory",
        "function",
        "developer",
        "critic",
        "reflection",
    ]
    content: str | list[ContentBlock]

    # 可选字段
    name: NotRequired[str]  # 发送者/工具名
    tool_call_id: NotRequired[str]  # 关联的 tool call（如 function 结果）
    metadata: NotRequired[dict[str, Any]]  # 附加信息
    timestamp: NotRequired[int]


# 所有 Message 的联合类型
Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | AgentMessage


__all__ = [
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "AgentMessage",
    "Message",
    "Cost",
    "Usage",
    "DiagnosticErrorInfo",
    "AssistantMessageDiagnostic",
    "ToolDetails",
]
