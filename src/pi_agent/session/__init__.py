"""pi_agent.session — Session 系统（Phase 3）。

DAG 会话树模型 + 持久化存储 + 会话搜索。
"""

from .jsonl import (
    JsonlSessionStorage,
    JsonlSessionStore,
    create_jsonl_session_repo,
    create_jsonl_session_store,
)
from .memory import (
    InMemorySessionStorage,
    InMemorySessionStore,
    create_in_memory_session_repo,
    create_in_memory_session_store,
)
from .repo import (
    SessionRepo,
    create_session_id,
    create_session_repo,
    create_timestamp,
    find_session_entry_matches,
    get_entries_to_fork,
    to_session,
    to_store_session,
)
from .search import ScanningSessionSearch, rebuild_session_search_index
from .session import (
    Session,
    build_context_entries,
    build_session_context,
    default_context_entry_transform,
)
from .types import (
    ActiveToolsChangeEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    JsonlSessionMetadata,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionContext,
    SessionEntryCursorOptions,
    SessionError,
    SessionInfoEntry,
    SessionMetadata,
    SessionSearch,
    SessionSearchHit,
    SessionSearchIndex,
    SessionSearchOptions,
    SessionSnapshot,
    SessionStats,
    SessionStorage,
    SessionStore,
    SessionTreeEntry,
    ThinkingLevelChangeEntry,
)

__all__ = [
    # Session 类
    "Session",
    "default_context_entry_transform",
    "build_context_entries",
    "build_session_context",
    # 存储
    "SessionStorage",
    "InMemorySessionStorage",
    "InMemorySessionStore",
    "JsonlSessionStorage",
    "JsonlSessionStore",
    "SessionStore",
    "SessionRepo",
    "create_in_memory_session_store",
    "create_in_memory_session_repo",
    "create_jsonl_session_store",
    "create_jsonl_session_repo",
    "create_session_repo",
    "to_session",
    "to_store_session",
    "create_session_id",
    "create_timestamp",
    "get_entries_to_fork",
    # 搜索
    "ScanningSessionSearch",
    "rebuild_session_search_index",
    "find_session_entry_matches",
    "SessionSearch",
    "SessionSearchIndex",
    "SessionSearchOptions",
    "SessionSearchHit",
    # 类型
    "SessionError",
    "SessionTreeEntry",
    "MessageEntry",
    "ThinkingLevelChangeEntry",
    "ModelChangeEntry",
    "ActiveToolsChangeEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "LabelEntry",
    "SessionInfoEntry",
    "LeafEntry",
    "SessionContext",
    "SessionStats",
    "SessionMetadata",
    "JsonlSessionMetadata",
    "SessionSnapshot",
    "SessionEntryCursorOptions",
]
