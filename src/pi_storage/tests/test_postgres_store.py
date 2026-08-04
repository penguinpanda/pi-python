"""PostgreSQL 会话存储测试。

需要可用 PG（默认 localhost:5432/pi_test，可用 PI_PG_DSN 覆盖）；
缺 PG 时整模块 skipif 跳过（路线图 P2-2 验收口径）。
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from pi_storage.migrations import MIGRATIONS, SCHEMA_VERSION
from pi_storage.store import PostgresSessionStore, create_session_id


def _dsn() -> str:
    return os.environ.get("PI_PG_DSN", "postgresql://pi:pi@localhost:5432/pi_test")


async def _probe() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(_dsn(), timeout=2)
        await conn.close()
        return True
    except Exception:
        return False


PG_AVAILABLE = asyncio.run(_probe())


@pytest.fixture
async def store():
    schema = f"test_{uuid.uuid4().hex[:10]}"
    instance = PostgresSessionStore(_dsn(), schema=schema)
    await instance.open()
    await instance.migrate()
    try:
        yield instance
    finally:
        async with instance.pool.acquire() as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await instance.close()


class TestMigrations:
    def test_schema_version_matches_migrations(self):
        assert SCHEMA_VERSION == len(MIGRATIONS) >= 1

    def test_initial_tables_present(self):
        sql = "\n".join(MIGRATIONS)
        for table in (
            "sessions",
            "session_entries",
            "session_sequences",
            "branch_entries",
            "session_materialized",
            "entry_materialized",
        ):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        assert "pg_trgm" in sql


class TestPostgresStore:
    pytestmark = pytest.mark.skipif(
        not PG_AVAILABLE,
        reason="PostgreSQL not available (set PI_PG_DSN to a reachable database)",
    )

    async def test_create_list_get_delete(self, store):
        meta = await store.create_session("/tmp/proj", metadata={"name": "x"})
        assert meta.cwd == "/tmp/proj"
        assert meta.metadata == {"name": "x"}

        listed = await store.list_sessions(cwd="/tmp/proj")
        assert [item.id for item in listed] == [meta.id]
        assert await store.list_sessions(cwd="/other") == []

        fetched = await store.get_session(meta.id)
        assert fetched is not None and fetched.id == meta.id

        await store.delete_session(meta.id)
        assert await store.get_session(meta.id) is None
        assert await store.list_sessions() == []

    async def test_append_and_get_entries(self, store):
        meta = await store.create_session("/tmp/proj")
        seq1 = await store.append_entry(
            meta.id,
            {
                "id": create_session_id(),
                "type": "message",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        )
        seq2 = await store.append_entry(
            meta.id,
            {
                "id": create_session_id(),
                "type": "message",
                "timestamp": "2026-01-01T00:00:01+00:00",
            },
        )
        assert seq2 == seq1 + 1
        entries = await store.get_entries(meta.id)
        assert [entry["id"] for entry in entries] == [
            (await store.get_entries(meta.id, since_seq=0))[0]["id"],
            entries[1]["id"],
        ]
        since = await store.get_entries(meta.id, since_seq=seq1)
        assert len(since) == 1

    async def test_leaf_id_and_branch(self, store):
        meta = await store.create_session("/tmp/proj")
        entry_id = create_session_id()
        await store.append_entry(
            meta.id,
            {"id": entry_id, "type": "message", "timestamp": "2026-01-01T00:00:00+00:00"},
            branch_id="b1",
        )
        await store.set_leaf_id(meta.id, entry_id)
        assert await store.get_leaf_id(meta.id) == entry_id
        branch = await store.get_branch(meta.id, "b1")
        assert [entry["id"] for entry in branch] == [entry_id]

    async def test_search(self, store):
        meta = await store.create_session("/tmp/proj")
        await store.append_entry(
            meta.id,
            {
                "id": create_session_id(),
                "type": "message",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "message": {"role": "user", "content": "fix the flaky pytest"},
            },
        )
        other = await store.create_session("/tmp/other")
        await store.append_entry(
            other.id,
            {
                "id": create_session_id(),
                "type": "message",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "message": {"role": "user", "content": "unrelated note"},
            },
        )
        results = await store.search_session_ids("flaky pytest")
        assert meta.id in results
        assert other.id not in results

    async def test_unknown_session_append_raises(self, store):
        with pytest.raises(LookupError):
            await store.append_entry(
                "missing",
                {"id": "e1", "type": "message", "timestamp": "2026-01-01T00:00:00+00:00"},
            )
