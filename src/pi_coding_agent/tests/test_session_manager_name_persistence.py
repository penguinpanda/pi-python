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


@pytest.mark.asyncio
async def test_close_waits_for_scheduled_name_persistence(tmp_path):
    """close() 等待 set_session_name 调度的持久化任务完成（不依赖 sleep(0) 让步）。"""
    manager = SessionManager.create(
        cwd=str(tmp_path / "project"),
        sessions_dir=str(tmp_path / "sessions"),
        session_id="closing",
    )
    manager.set_session_name("Flush Me")
    await manager.close()

    reopened = SessionManager.open(manager.session_path)
    assert reopened.session_name == "Flush Me"
    assert reopened.session_path is not None


@pytest.mark.asyncio
async def test_close_waits_for_scheduled_label_persistence(tmp_path):
    """close() 等待 set_label 调度的持久化任务完成后再释放 repo。"""
    manager = SessionManager.create(
        cwd=str(tmp_path / "project"),
        sessions_dir=str(tmp_path / "sessions"),
        session_id="labelled",
    )
    entry_id = await manager.append_message({"role": "user", "content": "hi", "timestamp": 1})
    manager.set_label(entry_id, "important")
    await manager.close()

    reopened = SessionManager.open(manager.session_path)
    assert reopened.get_label(entry_id) == "important"
