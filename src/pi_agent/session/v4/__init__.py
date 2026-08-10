"""JSONL v4 Session 内存模型（对齐 TS harness/session 的 v4 语义）。

本子包是 `docs/nd_upload/pi-agent/session-v4-migration-plan.md` M0 的产物：
只包含纯内存模型（类型 / SessionState / Session facade / InMemory 存储），
不涉及文件读写。JSONL v4 存储与迁移由后续里程碑实现。
"""

from .memory import InMemorySessionRepo, InMemorySessionStorage
from .repo import JsonlSessionRepo
from .search import ScanningSessionSearch
from .session import Session, SessionTree
from .state import SessionState
from .storage import JsonlSessionStorage
from .types import SessionError

__all__ = [
    "InMemorySessionRepo",
    "InMemorySessionStorage",
    "JsonlSessionRepo",
    "JsonlSessionStorage",
    "ScanningSessionSearch",
    "Session",
    "SessionError",
    "SessionState",
    "SessionTree",
]
