"""JSONL v4 存储类型（对齐 TS `harness/session/jsonl/types.ts`）。"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

from .fs import JsonlSessionRepoFileSystem
from .types import SessionCreateOptions, SessionMetadata


class JsonlSessionMetadata(SessionMetadata):
    """JSONL 会话元数据（含文件路径与来源格式）。"""

    cwd: str
    path: str
    modifiedAt: int
    sourceFormat: Literal[3, 4]
    legacyParentSessionPath: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class JsonlV4Header(TypedDict):
    """v4 会话文件首行。"""

    kind: Literal["header"]
    version: Literal[4]
    id: str
    createdAt: int
    cwd: str
    parentSessionId: NotRequired[str]
    legacyParentSessionPath: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class JsonlSessionCreateOptions(SessionCreateOptions, total=False):
    """创建 JSONL 会话的选项。"""

    cwd: str
    metadata: dict[str, Any]


class JsonlSessionListOptions(TypedDict, total=False):
    """列出 JSONL 会话的选项。"""

    cwd: str


class JsonlSessionRepoOptions(TypedDict):
    """JSONL 仓库构造选项（对齐 TS `JsonlSessionRepoOptions`）。"""

    fs: JsonlSessionRepoFileSystem
    sessionsRoot: str


__all__ = [
    "JsonlSessionMetadata",
    "JsonlV4Header",
    "JsonlSessionCreateOptions",
    "JsonlSessionListOptions",
    "JsonlSessionRepoOptions",
]
