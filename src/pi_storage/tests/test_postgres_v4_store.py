"""PostgreSQL v4 会话后端测试（需要可用 PG；缺 PG 时整模块 skip）。"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid

import pytest

from pi_storage.v4 import (
    PostgresV4SessionRepo,
    PgSessionSearch,
    _acquire_lease,
    _filter_entries,
)
from pi_storage.migrations import MIGRATIONS


def _dsn() -> str:
    return os.environ.get("PI_PG_DSN", "postgresql://pi:pi@localhost:5432/pi_test")


async def _probe() -> bool:
    # 先做 TCP 预检，避免无 PG 环境在 asyncpg 连接阶段长时间挂起。
    host, port_text = _dsn().split("@")[-1].rsplit(":", 1)
    port = int(port_text.split("/")[0])
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError:
        return False
    try:
        import asyncpg

        conn = await asyncpg.connect(_dsn(), timeout=2)
        await conn.close()
        return True
    except Exception:
        return False


PG_AVAILABLE = asyncio.run(_probe())


@pytest.fixture
async def repo():
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not available")
    schema = f"test_{uuid.uuid4().hex[:10]}"
    instance = PostgresV4SessionRepo(_dsn(), schema=schema)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.close()
        conn = await _dsn_connect()
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()


async def _dsn_connect():
    import asyncpg

    return await asyncpg.connect(_dsn())


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def _assistant(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "openai-completions",
        "provider": "faux",
        "model": "faux-1",
        "stopReason": "stop",
        "timestamp": 2,
    }


def _usage() -> dict:
    return {
        "input": 10,
        "output": 5,
        "cache_read": 3,
        "cache_write": 2,
        "total_tokens": 20,
        "cost": {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4, "total": 10},
    }


class TestMigrations:
    def test_v4_tables_present_in_migrations(self):
        sql = "\n".join(MIGRATIONS)
        for table in (
            "lanes",
            "records",
            "lane_moves",
            "facts",
            "branch_tips",
            "writer_leases",
            "session_stats",
        ):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        assert "ALTER TABLE branch_entries ADD COLUMN IF NOT EXISTS entry_type" in sql


class TestNonPgUnits:
    def test_repo_factory_construction(self):
        repo = PostgresV4SessionRepo("postgresql://x:x@localhost:5432/x")
        assert repo._dsn == "postgresql://x:x@localhost:5432/x"

    def test_filter_entries_cursor_and_limit(self):
        entries = [
            {"id": "1", "seq": 1, "type": "message"},
            {"id": "2", "seq": 2, "type": "custom", "customType": "note"},
            {"id": "3", "seq": 3, "type": "message"},
        ]
        filtered = _filter_entries(entries, {"type": "message"})
        assert [entry["id"] for entry in filtered] == ["1", "3"]
        limited = _filter_entries(
            entries,
            {"order": "oldestFirst", "cursor": {"afterSeq": 1}, "limit": 1},
        )
        assert [entry["id"] for entry in limited] == ["2"]


class TestCoreConformance:
    @pytest.mark.asyncio
    async def test_entries_lanes_records_facts_stats(self, repo):
        session = await repo.create({"id": "one", "cwd": "/tmp"})
        root = await session.append_entry(
            {"type": "message", "id": "root", "message": _user("root")}, "main"
        )
        await session.create_lane("thread", root["id"])
        child = await session.append_entry(
            {"type": "custom", "id": "child", "customType": "note", "data": {"v": 1}},
            "thread",
        )
        run = await session.append_record(
            {
                "type": "operation_started",
                "id": "run",
                "lane": "thread",
                "sourceLeafId": None,
                "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
            }
        )
        await session.set_name("Example")
        await session.set_label(root["id"], "checkpoint")
        await session.append_record(
            {
                "type": "usage",
                "id": "usage-1",
                "lane": "main",
                "cause": "assistant",
                "runId": "run",
                "entryId": "assistant",
                "attempt": 1,
                "stopReason": "stop",
                "usage": _usage(),
            }
        )

        assert (root["parentId"], root["seq"]) == (None, 1)
        assert (child["parentId"], child["seq"]) == ("root", 3)
        assert run["seq"] == 4
        assert await session.get_name() == "Example"
        assert await session.get_label("root") == "checkpoint"
        assert await session.get_lanes() == [
            {"lane": "main", "leafId": "root"},
            {"lane": "thread", "leafId": "child"},
        ]
        assert await session.get_stats() == {
            "messageCount": 1,
            "cachedTokens": 3,
            "uncachedTokens": 12,
            "totalTokens": 20,
            "costTotal": 10.0,
        }
        open_ops = await session.find_open_operations("thread")
        assert [op["id"] for op in open_ops] == ["run"]
        await session.append_record(
            {
                "type": "operation_finished",
                "id": "finish",
                "lane": "thread",
                "runId": "run",
                "outcome": "completed",
            }
        )
        assert await session.find_open_operations("thread") == []

        # 重新打开（复用 active storage），状态一致
        reopened = await repo.open(await session.get_metadata())
        assert await reopened.get_name() == "Example"
        assert len(await reopened.find_entries()) == 2

    @pytest.mark.asyncio
    async def test_duplicate_ids_rejected(self, repo):
        session = await repo.create({"id": "one", "cwd": "/tmp"})
        await session.append_entry(
            {"type": "message", "id": "shared", "message": _user("root")}, "main"
        )
        from pi_agent.session.v4.types import SessionError

        with pytest.raises(SessionError) as excinfo:
            await session.append_record(
                {
                    "type": "operation_started",
                    "id": "shared",
                    "lane": "main",
                    "sourceLeafId": None,
                    "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
                }
            )
        assert excinfo.value.code == "already_exists"

    @pytest.mark.asyncio
    async def test_branch_read_and_repair(self, repo):
        session = await repo.create({"id": "one", "cwd": "/tmp"})
        root = await session.append_message(_user("root"))
        main_child = await session.append_message(_user("main"))
        await session.create_lane("thread", root)
        thread_child = await session.view("thread").append_message(_user("thread"))

        branch_main = await session.find_entries_on_branch(
            {"start": main_child, "order": "oldestFirst"}
        )
        branch_thread = await session.find_entries_on_branch(
            {"start": thread_child, "order": "oldestFirst"}
        )
        assert [entry["id"] for entry in branch_main] == [root, main_child]
        assert [entry["id"] for entry in branch_thread] == [root, thread_child]

        await repo.repair_branch_cache(await session.get_metadata())
        branch_after = await session.find_entries_on_branch(
            {"start": main_child, "order": "oldestFirst"}
        )
        assert [entry["id"] for entry in branch_after] == [root, main_child]

    @pytest.mark.asyncio
    async def test_fork_branch_and_tree(self, repo):
        source = await repo.create({"id": "source", "cwd": "/tmp"})
        root = await source.append_message(_user("root"))
        shared = await source.append_message(_assistant("shared"))
        await source.create_lane("thread", shared)
        thread_child = await source.view("thread").append_message(_user("thread"))
        main_child = await source.append_message(_user("main"))
        await source.set_name("Source")
        await source.set_label(shared, "copied")

        fork = await repo.fork(
            await source.get_metadata(),
            {
                "scope": "branch",
                "entryId": main_child,
                "position": "at",
                "id": "fork",
                "cwd": "/tmp",
            },
        )
        assert [entry["id"] for entry in await fork.find_entries({"order": "oldestFirst"})] == [
            root,
            shared,
            main_child,
        ]
        assert await fork.get_name() == "Source"
        assert await fork.get_label(shared) == "copied"
        assert await fork.get_label(thread_child) is None

        tree = await repo.fork(
            await source.get_metadata(),
            {"scope": "tree", "id": "tree", "cwd": "/tmp"},
        )
        assert len(await tree.find_entries()) == 4
        assert await tree.get_lanes() == [
            {"lane": "main", "leafId": main_child},
            {"lane": "thread", "leafId": thread_child},
        ]

    @pytest.mark.asyncio
    async def test_list_delete_and_search(self, repo):
        session = await repo.create({"id": "a", "cwd": "/tmp/a"})
        await session.append_message(_user("needle entry"))
        await session.append_record(
            {
                "type": "usage",
                "id": "u1",
                "lane": "main",
                "cause": "assistant",
                "usage": _usage(),
            }
        )
        await session.set_name("needle name")
        await repo.create({"id": "b", "cwd": "/tmp/b"})

        listed = await repo.list({"cwd": "/tmp/a"})
        assert [item["id"] for item in listed] == ["a"]

        search = PgSessionSearch(repo)
        hits = await search.search({"text": "needle", "cwd": "/tmp/a"})
        assert len(hits) >= 2  # entry + fact 命中
        assert all(hit["metadata"]["id"] == "a" for hit in hits)

        await repo.delete(await session.get_metadata())
        assert [item["id"] for item in await repo.list()] == ["b"]


class TestWriterLease:
    @pytest.mark.asyncio
    async def test_lease_blocks_second_writer_and_releases(self, repo):
        await repo.create({"id": "one", "cwd": "/tmp"})

        conn = await repo._acquire()
        try:
            second = await _acquire_lease(conn, "one", "other-owner")
            assert second is None
        finally:
            await repo.pool.release(conn)

        storage = repo._active_storages["one"]
        await storage.release()
        conn = await repo._acquire()
        try:
            second = await _acquire_lease(conn, "one", "other-owner")
            assert second is not None
            assert second.owner_id == "other-owner"
        finally:
            await repo.pool.release(conn)

    @pytest.mark.asyncio
    async def test_manager_from_repo_and_close(self, repo):
        from pi_coding_agent._session_manager_v4 import V4SessionManager

        manager = await V4SessionManager.from_repo(repo, "/tmp", session_id="mgr")
        await manager.append_message(_user("hi"))
        assert [m["role"] for m in manager.build_context()] == ["user"]
        await manager.close()
