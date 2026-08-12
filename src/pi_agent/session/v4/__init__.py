"""v4 Session 系统（对齐 TS `harness/session`）。

包含内存模型、JSONL 持久化/迁移、搜索索引、reducer 与后端 conformance 工厂。
"""

from __future__ import annotations

from .fs import FileInfo, JsonlSessionRepoFileSystem, LocalFileSystem
from .jsonl_types import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlV4Header,
)
from .memory import InMemorySessionRepo, InMemorySessionStorage
from .repo import JsonlSessionRepo
from .reducer import (
    EffectiveLaneConfiguration,
    LaneReductionInput,
    LaneReductionResult,
    LaneState,
    RecordLogCorruption,
    RecordLogCorruptionReason,
    RecordLogSlice,
    TerminalFailureState,
    ToolBatchState,
    reduce_lane_state,
    validate_record_log,
)
from .search import ScanningSessionSearch
from .session import Session, SessionTree
from .state import SessionState
from .storage import JsonlSessionStorage
from .testing import (
    SessionBackendConformanceCase,
    SessionBackendFixture,
    SessionBackendFixtureFactory,
    create_session_backend_conformance,
)
from .types import SessionError

__all__ = [
    "FileInfo",
    "InMemorySessionRepo",
    "InMemorySessionStorage",
    "JsonlSessionRepo",
    "JsonlSessionRepoFileSystem",
    "JsonlSessionRepoOptions",
    "JsonlSessionStorage",
    "JsonlSessionCreateOptions",
    "JsonlSessionListOptions",
    "JsonlSessionMetadata",
    "JsonlV4Header",
    "LocalFileSystem",
    "EffectiveLaneConfiguration",
    "LaneReductionInput",
    "LaneReductionResult",
    "LaneState",
    "RecordLogCorruption",
    "RecordLogCorruptionReason",
    "RecordLogSlice",
    "ScanningSessionSearch",
    "SessionBackendConformanceCase",
    "SessionBackendFixture",
    "SessionBackendFixtureFactory",
    "Session",
    "SessionError",
    "SessionState",
    "SessionTree",
    "TerminalFailureState",
    "ToolBatchState",
    "create_session_backend_conformance",
    "reduce_lane_state",
    "validate_record_log",
]
