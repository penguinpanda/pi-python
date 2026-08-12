"""JSONL v4 后端 conformance 测试。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.testing.conformance import (
    SessionBackendConformanceCase,
    SessionBackendFixture,
    create_session_backend_conformance,
)


class _JsonlFixture:
    def __init__(self, repo: JsonlSessionRepo, root: Path) -> None:
        self._repo = repo
        self._root = root

    @property
    def repository(self) -> JsonlSessionRepo:
        return self._repo

    async def dispose(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)


def _conformance_cases() -> list[SessionBackendConformanceCase]:
    async def factory() -> SessionBackendFixture:
        root = Path(tempfile.mkdtemp(prefix="pi-jsonl-conformance-"))
        return _JsonlFixture(JsonlSessionRepo(root), root)

    return create_session_backend_conformance(factory)


@pytest.mark.parametrize(
    "case",
    _conformance_cases(),
    ids=lambda case: f"{case.group}: {case.name}",
)
@pytest.mark.asyncio
async def test_jsonl_conformance(case: SessionBackendConformanceCase) -> None:
    await case.run()
