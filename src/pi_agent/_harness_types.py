"""AgentHarness 类型定义（对齐 TS legacy harness，44289550a^ / 0.84.0 之前）。

包含：
- Result / ok / err 辅助；
- 泛型资源、工具与选项（Python 用 dict JSON Schema 代替 TypeBox）；
- provider payload / response hooks 与 compaction/branch-summary retry 事件；
- navigate 完整选项与 TreePreparation。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Literal, Protocol, TypeVar, TypedDict
from typing_extensions import NotRequired

from pi_ai import Models, RetryPolicy
from pi_ai.types import (
    CacheRetention,
    ImageContent,
    Model,
    Usage,
)

from ._types import (
    AgentEvent,
    AgentMessage,
    AgentToolResult,
    QueueMode,
    ThinkingLevel,
    ToolExecutionMode,
)
from .compaction import CompactionSettings
from .session import Session

# ---------------------------------------------------------------------------
# Result / 错误
# ---------------------------------------------------------------------------

AgentHarnessErrorCode = Literal[
    "busy",
    "invalid_state",
    "invalid_argument",
    "session",
    "hook",
    "auth",
    "compaction",
    "branch_summary",
    "unknown",
]


class AgentHarnessError(Exception):
    """AgentHarness 顶层错误，带稳定分类 code（对齐 TS AgentHarnessError）。"""

    def __init__(
        self,
        code: AgentHarnessErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Result(Generic[T, E]):
    """对齐 TS `Result<T, E>`：`{ok: true, value}` / `{ok: false, error}`。"""

    ok: bool
    value: T | None = None
    error: E | None = None

    def is_ok(self) -> bool:
        return self.ok

    def get_or_throw(self) -> T:
        if not self.ok:
            error = self.error
            if isinstance(error, BaseException):
                raise error
            raise AgentHarnessError("unknown", str(error or "Result failed"))
        assert self.value is not None
        return self.value

    def get_or_none(self) -> T | None:
        return self.value if self.ok else None


def ok(value: T) -> Result[T, Any]:
    return Result(ok=True, value=value)


def err(error: E) -> Result[Any, E]:
    return Result(ok=False, error=error)


# ---------------------------------------------------------------------------
# 资源：Skill / PromptTemplate
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Skill:
    """从 SKILL.md 或应用提供的技能。"""

    name: str
    description: str = ""
    content: str = ""
    file_path: str = ""
    disable_model_invocation: bool = False


@dataclass(slots=True)
class PromptTemplate:
    """显式调用时可格式化的提示模板。"""

    name: str
    content: str = ""
    description: str | None = None


TSkill = TypeVar("TSkill")
TPromptTemplate = TypeVar("TPromptTemplate")
TContext = TypeVar("TContext")
TContextIn = TypeVar("TContextIn", contravariant=True)


@dataclass(slots=True)
class AgentHarnessResources(Generic[TSkill, TPromptTemplate]):
    """提供给显式调用方法与 system-prompt 回调的资源。"""

    skills: list[TSkill] | None = None
    prompt_templates: list[TPromptTemplate] | None = None

    def clone(self) -> "AgentHarnessResources[TSkill, TPromptTemplate]":
        return AgentHarnessResources(
            skills=list(self.skills) if self.skills is not None else None,
            prompt_templates=(
                list(self.prompt_templates) if self.prompt_templates is not None else None
            ),
        )


class AgentHarnessTool(Protocol[TContextIn]):
    """context-aware 工具（对齐 TS AgentHarnessTool；schema 用 dict JSON Schema）。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    label: str
    execution_mode: ToolExecutionMode
    prompt_snippet: str | None
    before_execute: Callable[[dict[str, Any], Any], Awaitable[Any]] | None
    after_execute: Callable[[Any], Awaitable[Any]] | None

    def execute(
        self,
        tool_call_id: str,
        params: Any,
        signal: Any,
        on_update: Any,
        context: TContextIn,
    ) -> Awaitable[AgentToolResult]: ...


# ---------------------------------------------------------------------------
# StreamOptions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentHarnessStreamOptions:
    """harness 持有的 provider 请求选项，按 turn 快照。"""

    transport: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    cache_retention: CacheRetention | None = None

    def clone(self) -> "AgentHarnessStreamOptions":
        return AgentHarnessStreamOptions(
            transport=self.transport,
            timeout_ms=self.timeout_ms,
            max_retries=self.max_retries,
            max_retry_delay_ms=self.max_retry_delay_ms,
            headers=dict(self.headers) if self.headers else None,
            metadata=dict(self.metadata) if self.metadata else None,
            cache_retention=self.cache_retention,
        )


_MISSING = object()


@dataclass(slots=True)
class AgentHarnessStreamOptionsPatch:
    """流选项补丁。字段默认 _MISSING 表示"未提供"；None 表示删除。"""

    transport: Any = _MISSING
    timeout_ms: Any = _MISSING
    max_retries: Any = _MISSING
    max_retry_delay_ms: Any = _MISSING
    cache_retention: Any = _MISSING
    headers: Any = _MISSING
    metadata: Any = _MISSING


def apply_stream_options_patch(
    base: AgentHarnessStreamOptions,
    patch: AgentHarnessStreamOptionsPatch | None,
) -> AgentHarnessStreamOptions:
    """将补丁应用到 base，返回新对象（不修改入参）。"""
    result = base.clone()
    if patch is None:
        return result
    if patch.transport is not _MISSING:
        result.transport = patch.transport
    if patch.timeout_ms is not _MISSING:
        result.timeout_ms = patch.timeout_ms
    if patch.max_retries is not _MISSING:
        result.max_retries = patch.max_retries
    if patch.max_retry_delay_ms is not _MISSING:
        result.max_retry_delay_ms = patch.max_retry_delay_ms
    if patch.cache_retention is not _MISSING:
        result.cache_retention = patch.cache_retention
    if patch.headers is not _MISSING:
        if patch.headers is None:
            result.headers = None
        else:
            headers = dict(result.headers or {})
            for key, value in patch.headers.items():
                if value is None:
                    headers.pop(key, None)
                else:
                    headers[key] = value
            result.headers = headers or None
    if patch.metadata is not _MISSING:
        if patch.metadata is None:
            result.metadata = None
        else:
            metadata = dict(result.metadata or {})
            for key, value in patch.metadata.items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
            result.metadata = metadata or None
    return result


# ---------------------------------------------------------------------------
# 阶段 / 导航 / 树
# ---------------------------------------------------------------------------

AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]


@dataclass(slots=True)
class NavigateOptions:
    summarize: bool = False
    custom_instructions: str | None = None
    replace_instructions: bool = False
    label: str | None = None


@dataclass(slots=True)
class TreePreparation:
    target_id: str
    old_leaf_id: str | None
    common_ancestor_id: str | None = None
    entries_to_summarize: list[Any] = field(default_factory=list)
    user_wants_summary: bool = False
    custom_instructions: str | None = None
    replace_instructions: bool = False
    label: str | None = None


# ---------------------------------------------------------------------------
# Harness 事件
# ---------------------------------------------------------------------------


class QueueUpdateEvent(TypedDict):
    type: Literal["queue_update"]
    steer: list[AgentMessage]
    follow_up: list[AgentMessage]
    next_turn: list[AgentMessage]


class SavePointEvent(TypedDict):
    type: Literal["save_point"]
    had_pending_mutations: bool


class AbortEvent(TypedDict):
    type: Literal["abort"]
    cleared_steer: list[AgentMessage]
    cleared_follow_up: list[AgentMessage]


class SettledEvent(TypedDict):
    type: Literal["settled"]
    next_turn_count: int


class BeforeAgentStartEvent(TypedDict):
    type: Literal["before_agent_start"]
    prompt: str
    images: list[ImageContent] | None
    system_prompt: str
    resources: AgentHarnessResources


class ContextEvent(TypedDict):
    type: Literal["context"]
    messages: list[AgentMessage]


class BeforeProviderRequestEvent(TypedDict):
    type: Literal["before_provider_request"]
    model: Model
    session_id: str
    stream_options: AgentHarnessStreamOptions


class BeforeProviderPayloadEvent(TypedDict):
    type: Literal["before_provider_payload"]
    model: Model
    payload: Any


class AfterProviderResponseEvent(TypedDict):
    type: Literal["after_provider_response"]
    status: int
    headers: dict[str, str]


class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]


class ToolResultEvent(TypedDict):
    type: Literal["tool_result"]
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    content: list[Any]
    details: Any
    is_error: bool
    usage: Usage | None


class SessionBeforeCompactEvent(TypedDict):
    type: Literal["session_before_compact"]
    preparation: Any
    branch_entries: list[Any]
    custom_instructions: str | None
    signal: Any


class SessionCompactEvent(TypedDict):
    type: Literal["session_compact"]
    compaction_entry: Any
    from_hook: bool


class SessionBeforeTreeEvent(TypedDict):
    type: Literal["session_before_tree"]
    preparation: TreePreparation
    signal: Any


class SessionTreeEvent(TypedDict):
    type: Literal["session_tree"]
    new_leaf_id: str
    old_leaf_id: str | None
    summary_entry: NotRequired[Any]
    from_hook: bool


class RetryScheduledEvent(TypedDict):
    type: Literal["retry_scheduled"]
    operation: Literal["compaction", "branch_summary"]
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str


class RetryAttemptStartEvent(TypedDict):
    type: Literal["retry_attempt_start"]
    operation: Literal["compaction", "branch_summary"]


class RetryFinishedEvent(TypedDict):
    type: Literal["retry_finished"]
    operation: Literal["compaction", "branch_summary"]


class ModelUpdateEvent(TypedDict):
    type: Literal["model_update"]
    model: Model
    previous_model: Model | None
    source: Literal["set", "restore"]


class ThinkingLevelUpdateEvent(TypedDict):
    type: Literal["thinking_level_update"]
    level: ThinkingLevel
    previous_level: ThinkingLevel


class ToolsUpdateEvent(TypedDict):
    type: Literal["tools_update"]
    tool_names: list[str]
    previous_tool_names: list[str]
    active_tool_names: list[str]
    previous_active_tool_names: list[str]
    source: Literal["set", "restore"]


class ResourcesUpdateEvent(TypedDict):
    type: Literal["resources_update"]
    resources: AgentHarnessResources
    previous_resources: AgentHarnessResources


HarnessOwnEvent = (
    QueueUpdateEvent
    | SavePointEvent
    | AbortEvent
    | SettledEvent
    | BeforeAgentStartEvent
    | ContextEvent
    | BeforeProviderRequestEvent
    | BeforeProviderPayloadEvent
    | AfterProviderResponseEvent
    | ToolCallEvent
    | ToolResultEvent
    | SessionBeforeCompactEvent
    | SessionCompactEvent
    | SessionBeforeTreeEvent
    | SessionTreeEvent
    | RetryScheduledEvent
    | RetryAttemptStartEvent
    | RetryFinishedEvent
    | ModelUpdateEvent
    | ThinkingLevelUpdateEvent
    | ToolsUpdateEvent
    | ResourcesUpdateEvent
)

AgentHarnessEvent = AgentEvent | HarnessOwnEvent


# ---------------------------------------------------------------------------
# Hook 结果 / 事件结果表
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BeforeAgentStartResult:
    messages: list[AgentMessage] | None = None
    system_prompt: str | None = None


@dataclass(slots=True)
class ContextResult:
    messages: list[AgentMessage]


@dataclass(slots=True)
class BeforeProviderRequestResult:
    stream_options: AgentHarnessStreamOptionsPatch | None = None


@dataclass(slots=True)
class BeforeProviderPayloadResult:
    payload: Any


@dataclass(slots=True)
class ToolCallResult:
    block: bool = False
    reason: str = ""


@dataclass(slots=True)
class ToolResultPatch:
    content: list[Any] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class SessionBeforeCompactResult:
    cancel: bool = False
    compaction: "CompactResult | None" = None


@dataclass(slots=True)
class SessionBeforeTreeResult:
    cancel: bool = False
    summary: Any = None
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


class AgentHarnessEventResultMap(TypedDict, total=False):
    before_agent_start: BeforeAgentStartResult | None
    context: ContextResult | None
    before_provider_request: BeforeProviderRequestResult | None
    before_provider_payload: BeforeProviderPayloadResult | None
    after_provider_response: None
    tool_call: ToolCallResult | None
    tool_result: ToolResultPatch | None
    session_before_compact: SessionBeforeCompactResult | None
    session_compact: None
    session_before_tree: SessionBeforeTreeResult | None
    session_tree: None
    retry_scheduled: None
    retry_attempt_start: None
    retry_finished: None
    model_update: None
    thinking_level_update: None
    resources_update: None
    tools_update: None
    queue_update: None
    save_point: None
    abort: None
    settled: None


# ---------------------------------------------------------------------------
# 操作结果
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AbortResult:
    cleared_steer: list[AgentMessage]
    cleared_follow_up: list[AgentMessage]


@dataclass(slots=True)
class CompactResult:
    summary: str
    first_kept_entry_id: str | None = None
    tokens_before: int = 0
    usage: Usage | None = None
    retained_tail: list[AgentMessage] | None = None
    details: Any = None


@dataclass(slots=True)
class NavigateTreeResult:
    cancelled: bool
    editor_text: str | None = None
    summary_entry: Any = None  # BranchSummaryEntry


# ---------------------------------------------------------------------------
# AgentHarnessOptions
# ---------------------------------------------------------------------------

AgentHarnessSystemPrompt = str | Callable[[dict], str | Awaitable[str]]


@dataclass(slots=True)
class AgentHarnessOptions(Generic[TContext]):
    """AgentHarness 构造选项（对齐 TS legacy AgentHarnessOptionsBase）。"""

    model: Model
    session: Session
    models: Models
    system_prompt: AgentHarnessSystemPrompt | None = None
    tools: list[AgentHarnessTool[TContext]] | None = None
    active_tool_names: list[str] | None = None
    resources: AgentHarnessResources | None = None
    stream_options: AgentHarnessStreamOptions | None = None
    compaction_settings: CompactionSettings | None = None
    thinking_level: ThinkingLevel = "off"
    tool_context: TContext | Callable[[], TContext | Awaitable[TContext]] | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    retry: RetryPolicy | None = None


__all__ = [
    "AgentHarnessError",
    "AgentHarnessErrorCode",
    "AgentHarnessOptions",
    "AgentHarnessResources",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "AgentHarnessSystemPrompt",
    "AgentHarnessPhase",
    "AgentHarnessEvent",
    "AgentHarnessEventResultMap",
    "Result",
    "ok",
    "err",
    "Skill",
    "PromptTemplate",
    "AgentHarnessTool",
    "NavigateOptions",
    "TreePreparation",
    "QueueUpdateEvent",
    "SavePointEvent",
    "AbortEvent",
    "SettledEvent",
    "BeforeAgentStartEvent",
    "ContextEvent",
    "BeforeProviderRequestEvent",
    "BeforeProviderPayloadEvent",
    "AfterProviderResponseEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "SessionBeforeCompactEvent",
    "SessionCompactEvent",
    "SessionBeforeTreeEvent",
    "SessionTreeEvent",
    "RetryScheduledEvent",
    "RetryAttemptStartEvent",
    "RetryFinishedEvent",
    "ModelUpdateEvent",
    "ThinkingLevelUpdateEvent",
    "ToolsUpdateEvent",
    "ResourcesUpdateEvent",
    "BeforeAgentStartResult",
    "ContextResult",
    "BeforeProviderRequestResult",
    "BeforeProviderPayloadResult",
    "ToolCallResult",
    "ToolResultPatch",
    "SessionBeforeCompactResult",
    "SessionBeforeTreeResult",
    "AbortResult",
    "CompactResult",
    "NavigateTreeResult",
    "apply_stream_options_patch",
]
