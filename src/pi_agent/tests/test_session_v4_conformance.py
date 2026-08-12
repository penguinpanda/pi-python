"""v4 Session 内存后端 conformance 测试。

复用 `pi_agent.session.v4.testing` 的后端无关用例，backend 为
`InMemorySessionRepo`（对齐 TS `packages/agent/src/harness/session/testing`）。
"""

from __future__ import annotations

import pytest

from pi_agent.session.v4.memory import InMemorySessionRepo
from pi_agent.session.v4.testing.conformance import (
    SessionBackendConformanceCase,
    SessionBackendFixture,
    create_assistant_message,
    create_session_backend_conformance,
    create_user_message,
    entry_ids,
    operation_started,
    rejects_with_code,
)

__all__ = [
    "create_assistant_message",
    "create_user_message",
    "entry_ids",
    "operation_started",
    "rejects_with_code",
]


class _InMemoryFixture:
    def __init__(self, repository: InMemorySessionRepo) -> None:
        self._repository = repository

    @property
    def repository(self) -> InMemorySessionRepo:
        return self._repository

    async def dispose(self) -> None:
        return None


def _conformance_cases() -> list[SessionBackendConformanceCase]:
    async def factory() -> SessionBackendFixture:
        return _InMemoryFixture(InMemorySessionRepo())

    return create_session_backend_conformance(factory)


@pytest.mark.parametrize(
    "case",
    _conformance_cases(),
    ids=lambda case: f"{case.group}: {case.name}",
)
@pytest.mark.asyncio
async def test_in_memory_conformance(case: SessionBackendConformanceCase) -> None:
    await case.run()
