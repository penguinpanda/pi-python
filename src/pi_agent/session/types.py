"""Session 系统类型定义（Phase 3）。

对齐 TS `harness/types.ts` 的 session 部分。条目 JSON 键使用 camelCase，
与 TS JSONL 文件格式保持一致。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from typing_extensions import NotRequired

from pi_ai.types import ImageContent, TextContent, Usage

from .._types import AgentMessage


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------

SessionErrorCode = Literal[
    "not_found",
    "invalid_session",
    "invalid_entry",
    "invalid_fork_target",
    "storage",
    "unknown",
]


class SessionError(Exception):
    """Session 子系统错误（对齐 TS SessionError）。"""

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
# 会话树条目
# ---------------------------------------------------------------------------


class _SessionTreeEntryBase(TypedDict):
    id: str
    parentId: str | None
    timestamp: str


class MessageEntry(_SessionTreeEntryBase):
    type: Literal["message"]
    message: AgentMessage


class ThinkingLevelChangeEntry(_SessionTreeEntryBase):
    type: Literal["thinking_level_change"]
    thinkingLevel: str


class ModelChangeEntry(_SessionTreeEntryBase):
    type: Literal["model_change"]
    provider: str
    modelId: str


class ActiveToolsChangeEntry(_SessionTreeEntryBase):
    type: Literal["active_tools_change"]
    activeToolNames: list[str]


class CompactionEntry(_SessionTreeEntryBase, total=False):
    type: Literal["compaction"]
    summary: str
    firstKeptEntryId: NotRequired[str | None]
    tokensBefore: NotRequired[int]
    retainedTail: NotRequired[list[AgentMessage]]
    details: NotRequired[Any]
    usage: NotRequired[Usage]
    fromHook: NotRequired[bool]


class BranchSummaryEntry(_SessionTreeEntryBase, total=False):
    type: Literal["branch_summary"]
    fromId: str
    summary: str
    details: NotRequired[Any]
    usage: NotRequired[Usage]
    fromHook: NotRequired[bool]


class CustomEntry(_SessionTreeEntryBase, total=False):
    type: Literal["custom"]
    customType: str
    data: NotRequired[Any]


class CustomMessageEntry(_SessionTreeEntryBase, total=False):
    type: Literal["custom_message"]
    customType: str
    content: str | list[TextContent | ImageContent]
    display: bool
    details: NotRequired[Any]


class LabelEntry(_SessionTreeEntryBase):
    type: Literal["label"]
    targetId: str
    label: str | None


class SessionInfoEntry(_SessionTreeEntryBase, total=False):
    type: Literal["session_info"]
    name: NotRequired[str]


class LeafEntry(_SessionTreeEntryBase):
    type: Literal["leaf"]
    targetId: str | None


SessionTreeEntry = (
    MessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | ActiveToolsChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
    | LeafEntry
)


# ---------------------------------------------------------------------------
# 上下文 / 统计 / 元数据
# ---------------------------------------------------------------------------


class SessionContext(TypedDict):
    messages: list[AgentMessage]
    thinkingLevel: str
    model: dict[str, str] | None
    activeToolNames: list[str] | None


class SessionStats(TypedDict):
    messageCount: int
    cachedTokens: int
    uncachedTokens: int
    totalTokens: int
    costTotal: float


class SessionMetadata(TypedDict):
    id: str
    createdAt: str


class JsonlSessionMetadata(SessionMetadata):
    cwd: str
    path: str
    parentSessionPath: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class SessionEntryCursorOptions(TypedDict, total=False):
    afterEntrySeq: NotRequired[int]
    limit: NotRequired[int]


class SessionSnapshot(TypedDict, total=False):
    metadata: SessionMetadata
    leafId: str | None
    entries: list[SessionTreeEntry]


# ---------------------------------------------------------------------------
# 存储 / 仓库协议
# ---------------------------------------------------------------------------


class SessionStorage(Protocol):
    """会话持久化抽象（对齐 TS SessionStorage）。"""

    async def get_metadata(self) -> SessionMetadata: ...

    async def get_leaf_id(self) -> str | None: ...

    async def set_leaf_id(self, leaf_id: str | None) -> LeafEntry: ...

    async def create_entry_id(self) -> str: ...

    async def append_entry(self, entry: SessionTreeEntry) -> None: ...

    async def get_entry(self, entry_id: str) -> SessionTreeEntry | None: ...

    async def find_entries(self, entry_type: str) -> list[SessionTreeEntry]: ...

    async def get_label(self, entry_id: str) -> str | None: ...

    async def get_session_name(self) -> str | None: ...

    async def get_session_stats(self) -> SessionStats: ...

    async def get_path_to_root_or_compaction(
        self, leaf_id: str | None
    ) -> list[SessionTreeEntry]: ...

    async def get_entries(
        self, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]: ...


class SessionCreateOptions(TypedDict, total=False):
    id: NotRequired[str]


class SessionForkOptions(TypedDict, total=False):
    entryId: NotRequired[str]
    position: NotRequired[Literal["before", "at"]]
    id: NotRequired[str]
    # JsonlSessionStore fork 扩展字段（会话文件布局）。
    cwd: NotRequired[str]
    parentSessionPath: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class SessionStore(Protocol):
    """会话存储仓库接口（对齐 TS SessionStore）。"""

    async def create(self, options: SessionCreateOptions) -> SessionMetadata: ...

    async def load(self, metadata: SessionMetadata) -> SessionSnapshot: ...

    async def get_entries(
        self, metadata: SessionMetadata, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]: ...

    async def create_entry_id(self, metadata: SessionMetadata) -> str: ...

    async def append_entry(self, metadata: SessionMetadata, entry: SessionTreeEntry) -> None: ...

    async def set_leaf_id(self, metadata: SessionMetadata, leaf_id: str | None) -> LeafEntry: ...

    async def delete(self, metadata: SessionMetadata) -> None: ...

    async def list(self) -> list[SessionMetadata]: ...

    async def fork(
        self, source: SessionMetadata, options: SessionForkOptions
    ) -> SessionMetadata: ...


# ---------------------------------------------------------------------------
# 搜索协议
# ---------------------------------------------------------------------------


class SessionSearchOptions(TypedDict, total=False):
    text: str
    cwd: NotRequired[str]


class SessionSearchHit(TypedDict, total=False):
    metadata: SessionMetadata
    entryId: str
    timestamp: str
    snippet: NotRequired[str]
    score: NotRequired[float]


class SessionSearch(Protocol):
    """会话搜索查询接口。"""

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]: ...


class SessionSearchIndex(Protocol):
    """派生搜索索引维护接口（与查询接口分离）。"""

    async def upsert_entry(self, metadata: SessionMetadata, entry: SessionTreeEntry) -> None: ...

    async def replace_session(
        self, metadata: SessionMetadata, entries: list[SessionTreeEntry]
    ) -> None: ...

    async def delete_session(self, metadata: SessionMetadata) -> None: ...
