"""V4SessionManager（M3 应用层 v4 接线）测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pi_coding_agent._session_manager_v4 import V4SessionManager


def _user_message(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _assistant_message(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "deepseek",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "stopReason": "stop",
        "timestamp": 2,
    }


def _v3_lines(cwd: str) -> list[str]:
    return [
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "v3-session",
                "timestamp": "2026-08-10T00:00:00+00:00",
                "cwd": cwd,
            },
            ensure_ascii=False,
        )
        + "\n",
        json.dumps(
            {
                "type": "message",
                "id": "root",
                "parentId": None,
                "timestamp": "2026-08-10T00:00:01+00:00",
                "message": _user_message("root"),
            },
            ensure_ascii=False,
        )
        + "\n",
        json.dumps(
            {
                "type": "compaction",
                "id": "compact",
                "parentId": "root",
                "timestamp": "2026-08-10T00:00:02+00:00",
                "summary": "checkpoint",
                "firstKeptEntryId": "tail",
                "tokensBefore": 100,
            },
            ensure_ascii=False,
        )
        + "\n",
        json.dumps(
            {
                "type": "message",
                "id": "tail",
                "parentId": "compact",
                "timestamp": "2026-08-10T00:00:03+00:00",
                "message": _user_message("tail"),
            },
            ensure_ascii=False,
        )
        + "\n",
        json.dumps(
            {
                "type": "session_info",
                "id": "info",
                "parentId": None,
                "timestamp": "2026-08-10T00:00:04+00:00",
                "name": "Migrated",
            },
            ensure_ascii=False,
        )
        + "\n",
        json.dumps(
            {
                "type": "leaf",
                "id": "leaf",
                "parentId": None,
                "timestamp": "2026-08-10T00:00:05+00:00",
                "targetId": "tail",
            },
            ensure_ascii=False,
        )
        + "\n",
    ]


class TestV4SessionManager:
    @pytest.mark.asyncio
    async def test_create_append_reopen_roundtrip(self, tmp_path):
        cwd = str(tmp_path / "project")
        manager = await V4SessionManager.create(
            cwd,
            sessions_dir=str(tmp_path / "sessions"),
            session_id="v4-session",
        )
        await manager.append_message(_user_message("root"))
        await manager.append_message(_assistant_message("answer"))
        await manager.append_thinking_level_change("high")
        await manager.append_model_change("deepseek", "deepseek-v4-pro")
        manager.set_session_name("My V4")
        await asyncio.sleep(0)

        assert manager.is_persisted()
        assert manager.session_name == "My V4"
        assert [m["role"] for m in manager.build_context()] == [
            "user",
            "assistant",
        ]
        path = manager.session_path
        assert path is not None
        assert Path(path).read_text(encoding="utf-8").startswith('{"kind": "header", "version": 4')

        reopened = await V4SessionManager.open(path, cwd_override=cwd)
        assert reopened.session_name == "My V4"
        assert [m["role"] for m in reopened.build_context()] == [
            "user",
            "assistant",
        ]
        assert reopened.get_last_model_change() == (
            "deepseek",
            "deepseek-v4-pro",
        )

    @pytest.mark.asyncio
    async def test_open_v3_file_lazily_converts(self, tmp_path):
        cwd = str(tmp_path / "project")
        filepath = tmp_path / "v3.jsonl"
        filepath.write_text("".join(_v3_lines(cwd)), encoding="utf-8")

        manager = await V4SessionManager.open(filepath, cwd_override=cwd)

        assert manager.session_name == "Migrated"
        assert manager.get_leaf_id() == "tail"
        roles = [m["role"] for m in manager.build_context()]
        assert roles[0] == "compactionSummary"
        assert "user" in roles
        assert Path(f"{filepath}.bak").exists()

    @pytest.mark.asyncio
    async def test_in_memory_mode(self):
        manager = await V4SessionManager.in_memory(".")
        await manager.append_message(_user_message("hi"))
        assert not manager.is_persisted()
        assert [m["role"] for m in manager.build_context()] == ["user"]

    @pytest.mark.asyncio
    async def test_compact_move_label_and_tree(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path),
            sessions_dir=str(tmp_path / "sessions"),
            session_id="flow",
        )
        root = await manager.append_message(_user_message("q1"))
        await manager.append_message(_assistant_message("a1"))
        await manager.append_compaction("checkpoint", root, 100)
        await manager.append_message(_user_message("q2"))
        manager.set_label(root, "start")
        await asyncio.sleep(0)

        assert manager.get_label(root) == "start"
        assert manager.build_context()[0]["role"] == "compactionSummary"
        tree = manager.get_tree()
        assert len(tree) == 1

        await manager.move_to(root)
        assert manager.get_leaf_id() == root

    @pytest.mark.asyncio
    async def test_fork_creates_v4_branch(self, tmp_path):
        manager = await V4SessionManager.create(
            str(tmp_path),
            sessions_dir=str(tmp_path / "sessions"),
            session_id="source",
        )
        root = await manager.append_message(_user_message("root"))
        await manager.append_message(_user_message("tail"))

        fork = await manager.fork(root, position="at", session_id="forked")

        assert fork.get_leaf_id() == root
        assert [e["id"] for e in fork.get_entries()] == [root]
        assert fork.session_path is not None
        assert (
            Path(fork.session_path)
            .read_text(encoding="utf-8")
            .startswith('{"kind": "header", "version": 4')
        )

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        root = str(tmp_path / "sessions")
        await V4SessionManager.create(
            str(tmp_path / "p1"),
            sessions_dir=root,
            session_id="one",
        )
        await V4SessionManager.create(
            str(tmp_path / "p2"),
            sessions_dir=root,
            session_id="two",
        )

        listed = await V4SessionManager.list_sessions(root)
        assert {item.session_id for item in listed} == {"one", "two"}
        p1 = await V4SessionManager.list_sessions(root, cwd=str(tmp_path / "p1"))
        assert [item.session_id for item in p1] == ["one"]
