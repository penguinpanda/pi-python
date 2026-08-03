"""AgentHarness 类型定义（Phase 2 骨架）。

对齐 TS `harness/types.ts` 中 Phase 2 需要的部分：错误分类、资源
（Skill / PromptTemplate）、StreamOptions（+Patch）、harness 事件、
hook 结果与 AgentHarnessOptions。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypedDict

from pi_ai.types import ImageContent, Model, TextContent, Usage

from ._types import AgentEvent, AgentMessage, AgentTool, QueueMode, StreamFn, ThinkingLevel

# ---------------------------------------------------------------------------
# 错误
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
    "not_implemented",
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


@dataclass(slots=True)
class AgentHarnessResources:
    """提供给显式调用方法与 system-prompt 回调的资源。"""
    skills: list[Skill] | None = None
    prompt_templates: list[PromptTemplate] | None = None

    def clone(self) -> "AgentHarnessResources":
        return AgentHarnessResources(
            skills=list(self.skills) if self.skills is not None else None,
            prompt_templates=(
                list(self.prompt_templates) if self.prompt_templates is not None else None
            ),
        )


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
    cache_retention: str | None = None

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
# 阶段
# ---------------------------------------------------------------------------

AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]


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


class SessionBeforeTreeEvent(TypedDict):
    type: Literal["session_before_tree"]
    target_id: str
    old_leaf_id: str
    summarize: bool
    custom_instructions: str | None
    label: str | None


class SessionCompactEvent(TypedDict):
    type: Literal["session_compact"]
    from_hook: bool


class SessionTreeEvent(TypedDict):
    type: Literal["session_tree"]
    new_leaf_id: str
    old_leaf_id: str
    from_hook: bool


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
    | ToolCallEvent
    | ToolResultEvent
    | SessionBeforeCompactEvent
    | SessionCompactEvent
    | SessionBeforeTreeEvent
    | SessionTreeEvent
    | ModelUpdateEvent
    | ThinkingLevelUpdateEvent
    | ToolsUpdateEvent
    | ResourcesUpdateEvent
)

AgentHarnessEvent = AgentEvent | HarnessOwnEvent


# ---------------------------------------------------------------------------
# Hook 结果
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
    summary_entry: Any = None  # BranchSummaryEntry（Phase 3）


# ---------------------------------------------------------------------------
# AgentHarnessOptions
# ---------------------------------------------------------------------------

AgentHarnessSystemPrompt = str | Callable[[dict], str]


@dataclass(slots=True)
class AgentHarnessOptions:
    """AgentHarness 构造选项（Phase 2 骨架）。"""
    model: Model
    session: Any | None = None  # 最小 Session；None 时自动创建内存会话
    system_prompt: AgentHarnessSystemPrompt | None = None
    tools: list[AgentTool] | None = None
    active_tool_names: list[str] | None = None
    resources: AgentHarnessResources | None = None
    stream_fn: StreamFn | None = None
    stream_options: AgentHarnessStreamOptions | None = None
    thinking_level: ThinkingLevel = "off"
    tool_context: Any = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
