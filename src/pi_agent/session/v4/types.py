"""JSONL v4 Session 类型（对齐 TS `harness/session/types.ts`）。

条目 / 记录 / 日志使用 camelCase 键，与 TS v4 格式一致；
`Usage` 复用 `pi_ai.types.Usage`（Python 侧为 snake_case 字段）。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from typing_extensions import NotRequired

from pi_ai.types import Usage

from ..._types import AgentMessage

# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------

SessionErrorCode = Literal[
    "not_found",
    "already_exists",
    "invalid_entry",
    "invalid_payload",
    "invalid_lane",
    "invalid_query",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    """v4 Session 子系统错误（对齐 TS SessionError）。"""

    def __init__(
        self,
        code: SessionErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


# ---------------------------------------------------------------------------
# 条目
# ---------------------------------------------------------------------------


class EntryBase(TypedDict):
    id: str
    seq: int
    parentId: str | None
    timestamp: int


class MessageEntry(EntryBase):
    type: Literal["message"]
    message: AgentMessage
    terminate: NotRequired[bool]


class ModelChangeEntry(EntryBase):
    type: Literal["model_change"]
    provider: str
    modelId: str


class ThinkingLevelEntry(EntryBase):
    type: Literal["thinking_level_change"]
    thinkingLevel: str


class ActiveToolsEntry(EntryBase):
    type: Literal["active_tools_change"]
    activeToolNames: list[str]


class CompactionEntry(EntryBase):
    type: Literal["compaction"]
    summary: str
    retainedTail: list[AgentMessage]
    tokensBefore: int
    details: NotRequired[Any]
    usage: NotRequired[Usage]


class BranchSummaryEntry(EntryBase):
    type: Literal["branch_summary"]
    fromId: str
    summary: str
    details: NotRequired[Any]
    usage: NotRequired[Usage]


class CustomEntry(EntryBase):
    type: Literal["custom"]
    customType: str
    data: NotRequired[Any]


Entry = (
    MessageEntry
    | ModelChangeEntry
    | ThinkingLevelEntry
    | ActiveToolsEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
)

# TS 的 ProvisionedEntry<T> = Omit<T, "parentId" | "seq" | "timestamp">。
# Python TypedDict 不支持条件 Omit，统一用宽松字典承载待写入条目。
ProvisionedEntry = dict[str, Any]


# ---------------------------------------------------------------------------
# 审计记录
# ---------------------------------------------------------------------------


class RecordBase(TypedDict):
    id: str
    seq: int
    lane: str
    timestamp: int


class RunIntent(TypedDict):
    kind: Literal["run"]
    originalPrompt: list[AgentMessage]
    initialMessages: list[ProvisionedEntry]
    systemPromptOverride: NotRequired[str]
    resumeData: NotRequired[dict[str, Any]]


class CompactionIntent(TypedDict):
    kind: Literal["compaction"]
    customInstructions: NotRequired[str]
    resultEntryId: str


class NavigationIntent(TypedDict):
    kind: Literal["navigation"]
    targetId: str | None
    summarize: bool
    customInstructions: NotRequired[str]
    label: NotRequired[str]
    summaryEntryId: NotRequired[str]


OperationIntent = RunIntent | CompactionIntent | NavigationIntent


class OperationStartedRecord(RecordBase):
    type: Literal["operation_started"]
    sourceLeafId: str | None
    intent: OperationIntent


class AbortRequestedRecord(RecordBase):
    type: Literal["abort_requested"]
    runId: str


class OperationFinishedRecord(RecordBase):
    type: Literal["operation_finished"]
    runId: str
    outcome: Literal["completed", "aborted", "failed", "declined"]
    error: NotRequired[dict[str, str]]


CompactionReason = Literal["manual", "threshold", "overflow"]


class AssistantStepAttemptRecord(RecordBase):
    type: Literal["step_attempt"]
    runId: str
    step: Literal["assistant", "branch_summary"]
    attempt: int
    resultEntryId: str


class CompactionStepAttemptRecord(RecordBase):
    type: Literal["step_attempt"]
    runId: str
    step: Literal["compaction"]
    attempt: int
    resultEntryId: str
    compactionReason: CompactionReason


StepAttemptRecord = AssistantStepAttemptRecord | CompactionStepAttemptRecord


class ToolStartedRecord(RecordBase):
    type: Literal["tool_started"]
    runId: str
    assistantEntryId: str
    toolIndex: int
    toolCallId: str
    toolName: str
    effectiveArgs: dict[str, Any]
    resultEntryId: str
    replay: Literal["never", "safe"]


class QueueEnqueuedRecord(RecordBase):
    type: Literal["queue_enqueued"]
    queue: Literal["steer", "followUp", "nextRun"]
    runId: NotRequired[str]
    target: ProvisionedEntry


class QueueCancelledRecord(RecordBase):
    type: Literal["queue_cancelled"]
    runId: NotRequired[str]
    entryId: str


class WriteDeferredRecord(RecordBase):
    type: Literal["write_deferred"]
    runId: str
    target: ProvisionedEntry


class UsageRecord(RecordBase):
    type: Literal["usage"]
    cause: Literal[
        "assistant", "compaction", "branch_summary", "deferred_fetch", "tool", "hook", "adjustment"
    ]
    runId: NotRequired[str]
    entryId: NotRequired[str]
    attempt: NotRequired[int]
    stopReason: NotRequired[str]
    toolCallId: NotRequired[str]
    details: NotRequired[Any]
    usage: Usage


LaneRecord = (
    OperationStartedRecord
    | AbortRequestedRecord
    | OperationFinishedRecord
    | StepAttemptRecord
    | ToolStartedRecord
    | QueueEnqueuedRecord
    | QueueCancelledRecord
    | WriteDeferredRecord
    | UsageRecord
)

# TS 的 NewRecord<T> = Omit<T, "seq" | "timestamp">；Python 侧统一用宽松字典。
NewRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# 查询 / 日志 / 元数据
# ---------------------------------------------------------------------------


class EntryCursor(TypedDict):
    afterSeq: int


class EntryQuery(TypedDict, total=False):
    type: str
    customType: str
    order: Literal["newestFirst", "oldestFirst"]
    limit: int
    cursor: EntryCursor


class BranchBounds(TypedDict, total=False):
    start: str
    stopAtType: str
    stopAtId: str


class BranchEntryQuery(EntryQuery, BranchBounds, total=False):
    """分支路径查询：EntryQuery + BranchBounds 的合取。"""

    start: NotRequired[str]


class RecordQuery(TypedDict, total=False):
    lane: str
    type: str
    runId: str
    operationKind: Literal["run", "compaction", "navigation"]
    afterSeq: int
    order: Literal["newestFirst", "oldestFirst"]
    limit: int


class LogOptions(TypedDict, total=False):
    afterSeq: int
    limit: int


class LanePointer(TypedDict):
    lane: str
    leafId: str | None


class SessionStats(TypedDict):
    messageCount: int
    cachedTokens: int
    uncachedTokens: int
    totalTokens: int
    costTotal: float


class SessionContext(TypedDict):
    """由会话路径投影出的 LLM 就绪上下文（对齐 TS SessionContext）。"""

    messages: list[AgentMessage]
    thinkingLevel: str
    model: dict[str, str] | None
    activeToolNames: list[str] | None


class SessionMetadata(TypedDict):
    id: str
    createdAt: int
    parentSessionId: NotRequired[str]


class SessionCreateOptions(TypedDict, total=False):
    id: str
    parentSessionId: str


class SessionSearchOptions(TypedDict, total=False):
    """会话搜索选项（对齐 TS SessionSearchOptions）。"""

    text: str
    cwd: str


class SessionSearchHit(TypedDict, total=False):
    """会话搜索命中（对齐 TS SessionSearchHit）。"""

    metadata: SessionMetadata
    entryId: str
    timestamp: str
    snippet: str
    score: float


class ForkOptions(SessionCreateOptions, total=False):
    scope: Literal["branch", "tree"]
    entryId: str
    position: Literal["before", "at"]


class EntryLogItem(TypedDict):
    kind: Literal["entry"]
    seq: int
    entry: Entry


class RecordLogItem(TypedDict):
    kind: Literal["record"]
    seq: int
    record: LaneRecord


class LaneLogItem(TypedDict):
    kind: Literal["lane"]
    seq: int
    lane: str
    leafId: str | None


class NameFactLogItem(TypedDict):
    kind: Literal["fact"]
    seq: int
    fact: Literal["name"]
    name: str


class LabelFactLogItem(TypedDict):
    kind: Literal["fact"]
    seq: int
    fact: Literal["label"]
    targetId: str
    label: str | None


LogItem = EntryLogItem | RecordLogItem | LaneLogItem | NameFactLogItem | LabelFactLogItem


# ---------------------------------------------------------------------------
# 存储 / 树 / 仓库协议
# ---------------------------------------------------------------------------


class SessionStorage(Protocol):
    """v4 会话持久化抽象（对齐 TS SessionStorage）。"""

    async def get_metadata(self) -> SessionMetadata: ...

    async def get_lanes(self) -> list[LanePointer]: ...

    async def create_lane(self, lane: str, at: str | None) -> None: ...

    async def move_lane(self, lane: str, to: str | None) -> None: ...

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry: ...

    async def append_record(self, record: NewRecord) -> LaneRecord: ...

    async def get_entry(self, entry_id: str) -> Entry | None: ...

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...

    async def find_entries_on_branch(
        self, query: BranchEntryQuery | None = None
    ) -> list[Entry]: ...

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]: ...

    async def find_open_operations(
        self, lane: str, options: dict[str, int] | None = None
    ) -> list[OperationStartedRecord]: ...

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]: ...

    async def get_name(self) -> str | None: ...

    async def set_name(self, name: str) -> None: ...

    async def get_label(self, entry_id: str) -> str | None: ...

    async def set_label(self, entry_id: str, label: str | None) -> None: ...

    async def get_stats(self) -> SessionStats: ...


class SessionTree(Protocol):
    """按 lane 绑定的会话视图（对齐 TS SessionTree）。"""

    async def get_leaf_id(self) -> str | None: ...

    async def get_entry(self, entry_id: str) -> Entry | None: ...

    async def get_stats(self) -> SessionStats: ...

    async def get_name(self) -> str | None: ...

    async def set_name(self, name: str) -> None: ...

    async def get_label(self, target_id: str) -> str | None: ...

    async def set_label(self, target_id: str, label: str | None) -> None: ...

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None: ...

    async def find_entries_on_branch(
        self, query: BranchEntryQuery | None = None
    ) -> list[Entry]: ...

    async def find_entry_on_branch(self, query: BranchEntryQuery | None = None) -> Entry | None: ...

    async def append_message(self, message: AgentMessage) -> str: ...

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str: ...


class SessionRepo(Protocol):
    """v4 会话仓库（对齐 TS SessionRepo）。"""

    async def create(self, options: SessionCreateOptions | None = None) -> Any: ...

    async def open(self, metadata: SessionMetadata) -> Any: ...

    async def list(self, options: Any = None) -> list[SessionMetadata]: ...

    async def delete(self, metadata: SessionMetadata) -> None: ...

    async def fork(self, source: SessionMetadata, options: ForkOptions | None = None) -> Any: ...


__all__ = [
    "SessionError",
    "SessionErrorCode",
    "Entry",
    "MessageEntry",
    "ModelChangeEntry",
    "ThinkingLevelEntry",
    "ActiveToolsEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "CustomEntry",
    "ProvisionedEntry",
    "OperationIntent",
    "OperationStartedRecord",
    "AbortRequestedRecord",
    "OperationFinishedRecord",
    "StepAttemptRecord",
    "ToolStartedRecord",
    "QueueEnqueuedRecord",
    "QueueCancelledRecord",
    "WriteDeferredRecord",
    "UsageRecord",
    "LaneRecord",
    "NewRecord",
    "EntryCursor",
    "EntryQuery",
    "BranchBounds",
    "BranchEntryQuery",
    "RecordQuery",
    "LogOptions",
    "LanePointer",
    "SessionStats",
    "SessionContext",
    "SessionMetadata",
    "SessionCreateOptions",
    "SessionSearchOptions",
    "SessionSearchHit",
    "ForkOptions",
    "LogItem",
    "SessionStorage",
    "SessionTree",
    "SessionRepo",
]
