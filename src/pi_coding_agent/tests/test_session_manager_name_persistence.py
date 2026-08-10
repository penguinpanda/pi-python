"""SessionManager 会话名持久化测试（P2）。"""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent._session_manager import SessionManager


@pytest.mark.asyncio
async def test_set_session_name_persists_and_restores(tmp_path):
    cwd = str(tmp_path / "project")
    manager = SessionManager.create(
        cwd,
        sessions_dir=str(tmp_path / "sessions"),
        session_id="named",
    )
    manager.set_session_name("My Task")
    await asyncio.sleep(0)

    reopened = SessionManager.open(manager.session_path)
    assert reopened.session_name == "My Task"
    assert reopened.session_path is not None


@pytest.mark.asyncio
async def test_set_session_name_in_memory_does_not_crash():
    manager = SessionManager.in_memory(cwd=".")
    manager.set_session_name("Memory")
    await asyncio.sleep(0)
    assert manager.session_name == "Memory"
