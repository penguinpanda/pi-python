"""pi_agent.session — v4 Session 公开 API。

旧 v3 会话实现已移除；本包只导出 `pi_agent.session.v4`。
"""

from __future__ import annotations

from .v4.context import build_session_context
from .v4.memory import InMemorySessionRepo, InMemorySessionStorage
from .v4.repo import JsonlSessionRepo
from .v4.search import ScanningSessionSearch
from .v4.session import Session, SessionTree
from .v4.storage import JsonlSessionStorage
from .v4.types import (
    SessionError,
    SessionRepo,
    SessionStorage,
)

__all__ = [
    "Session",
    "SessionTree",
    "SessionError",
    "SessionStorage",
    "SessionRepo",
    "InMemorySessionRepo",
    "InMemorySessionStorage",
    "JsonlSessionRepo",
    "JsonlSessionStorage",
    "ScanningSessionSearch",
    "build_session_context",
]
