"""PostgreSQL SessionStore / SessionSearch（接口对齐 TS sqlite-node）。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import asyncpg

from .migrations import apply_migrations

DEFAULT_DSN = "postgresql://pi:pi@localhost:5432/pi"


@dataclass(slots=True)
class SessionMetadata:
    """会话元数据（对齐 TS SqliteSessionMetadata 字段）。"""

    id: str
    created_at: str
    cwd: str
    parent_session_id: str | None = None
    metadata: dict | None = None
    active_leaf_id: str | None = None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "SessionMetadata":
        metadata = row.get("metadata")
        if isinstance(metadata, str) and metadata:
            try:
                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = None
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            cwd=row["cwd"],
            parent_session_id=row.get("parent_session_id"),
            metadata=metadata,
            active_leaf_id=row.get("active_leaf_id"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "cwd": self.cwd,
            "parentSessionId": self.parent_session_id,
            "metadata": self.metadata,
            "activeLeafId": self.active_leaf_id,
        }


def create_session_id() -> str:
    return uuid.uuid4().hex


class PostgresSessionStore:
    """基于 asyncpg 池的会话存储。

    schema 参数用于测试隔离：连接后执行 `SET search_path` 到独立 schema。
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn or DEFAULT_DSN
        self._schema = schema
        self._pool: asyncpg.Pool | None = None
        self._min_size = min_size
        self._max_size = max_size

    async def open(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        if self._schema:
            await self._pool.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Store is not open; call open() first")
        return self._pool

    async def migrate(self) -> int:
        """应用迁移（schema 隔离时在独立 schema 内执行）。"""
        if self._schema:
            async with self.pool.acquire() as conn:
                await conn.execute(f'SET search_path TO "{self._schema}", public')
                return await apply_migrations(conn)
        async with self.pool.acquire() as conn:
            return await apply_migrations(conn)

    async def _acquire(self):
        conn = await self.pool.acquire()
        if self._schema:
            await conn.execute(f'SET search_path TO "{self._schema}", public')
        return conn

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        cwd: str,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        metadata: dict | None = None,
    ) -> SessionMetadata:
        sid = session_id or create_session_id()
        created_at = _now_iso()
        conn = await self._acquire()
        try:
            await conn.execute(
                """
                INSERT INTO sessions (id, created_at, cwd, parent_session_id, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                sid,
                created_at,
                cwd,
                parent_session_id,
                json.dumps(metadata) if metadata is not None else None,
            )
            await conn.execute(
                "INSERT INTO session_sequences (session_id, next_seq) VALUES ($1, 1)",
                sid,
            )
        finally:
            await self.pool.release(conn)
        return SessionMetadata(
            id=sid,
            created_at=created_at,
            cwd=cwd,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )

    async def list_sessions(self, cwd: str | None = None) -> list[SessionMetadata]:
        conn = await self._acquire()
        try:
            if cwd is not None:
                rows = await conn.fetch(
                    "SELECT * FROM sessions WHERE cwd = $1 ORDER BY created_at DESC",
                    cwd,
                )
            else:
                rows = await conn.fetch("SELECT * FROM sessions ORDER BY created_at DESC")
            return [SessionMetadata.from_row(row) for row in rows]
        finally:
            await self.pool.release(conn)

    async def get_session(self, session_id: str) -> SessionMetadata | None:
        conn = await self._acquire()
        try:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
            return SessionMetadata.from_row(row) if row is not None else None
        finally:
            await self.pool.release(conn)

    async def delete_session(self, session_id: str) -> None:
        conn = await self._acquire()
        try:
            async with conn.transaction():
                for table in (
                    "branch_entries",
                    "session_entries",
                    "entry_materialized",
                    "session_materialized",
                    "session_sequences",
                ):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE session_id = $1",
                        session_id,
                    )
                await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
        finally:
            await self.pool.release(conn)

    async def set_leaf_id(self, session_id: str, leaf_id: str | None) -> str | None:
        conn = await self._acquire()
        try:
            await conn.execute(
                "UPDATE sessions SET active_leaf_id = $1 WHERE id = $2",
                leaf_id,
                session_id,
            )
            return leaf_id
        finally:
            await self.pool.release(conn)

    async def get_leaf_id(self, session_id: str) -> str | None:
        conn = await self._acquire()
        try:
            return await conn.fetchval(
                "SELECT active_leaf_id FROM sessions WHERE id = $1",
                session_id,
            )
        finally:
            await self.pool.release(conn)

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------

    async def append_entry(
        self,
        session_id: str,
        entry: dict,
        *,
        branch_id: str | None = None,
    ) -> int:
        """追加条目并返回分配的 entry_seq。"""
        conn = await self._acquire()
        try:
            async with conn.transaction():
                next_seq = await conn.fetchval(
                    """
                    UPDATE session_sequences SET next_seq = next_seq + 1
                    WHERE session_id = $1 RETURNING next_seq
                    """,
                    session_id,
                )
                if next_seq is None:
                    raise LookupError(f"Session not found: {session_id}")
                entry_seq = next_seq - 1
                await conn.execute(
                    """
                    INSERT INTO session_entries (
                        session_id, id, entry_seq, parent_id, type, timestamp, payload
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    session_id,
                    entry["id"],
                    entry_seq,
                    entry.get("parentId"),
                    entry.get("type", "message"),
                    entry.get("timestamp", _now_iso()),
                    json.dumps(entry, ensure_ascii=False),
                )
                branch = branch_id or session_id
                await conn.execute(
                    """
                    INSERT INTO branch_entries (
                        session_id, branch_id, entry_id, entry_seq
                    ) VALUES ($1, $2, $3, $4)
                    """,
                    session_id,
                    branch,
                    entry["id"],
                    entry_seq,
                )
            return entry_seq
        finally:
            await self.pool.release(conn)

    async def get_entries(self, session_id: str, *, since_seq: int | None = None) -> list[dict]:
        conn = await self._acquire()
        try:
            if since_seq is not None:
                rows = await conn.fetch(
                    """
                    SELECT payload FROM session_entries
                    WHERE session_id = $1 AND entry_seq > $2
                    ORDER BY entry_seq
                    """,
                    session_id,
                    since_seq,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT payload FROM session_entries
                    WHERE session_id = $1 ORDER BY entry_seq
                    """,
                    session_id,
                )
            return [json.loads(row["payload"]) for row in rows]
        finally:
            await self.pool.release(conn)

    async def get_branch(self, session_id: str, branch_id: str | None = None) -> list[dict]:
        branch = branch_id or session_id
        conn = await self._acquire()
        try:
            rows = await conn.fetch(
                """
                SELECT e.payload FROM branch_entries b
                JOIN session_entries e ON e.session_id = b.session_id AND e.id = b.entry_id
                WHERE b.session_id = $1 AND b.branch_id = $2
                ORDER BY b.entry_seq
                """,
                session_id,
                branch,
            )
            return [json.loads(row["payload"]) for row in rows]
        finally:
            await self.pool.release(conn)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(self, query: str, *, limit: int = 50) -> list[tuple[str, float]]:
        """全文/模糊搜索会话（tsvector + pg_trgm），返回 [(session_id, score)]。"""
        conn = await self._acquire()
        try:
            rows = await conn.fetch(
                """
                SELECT session_id,
                       GREATEST(
                           ts_rank(to_tsvector('simple', payload), plainto_tsquery('simple', $1)),
                           similarity(payload, $1)
                       ) AS score
                FROM session_entries
                WHERE to_tsvector('simple', payload) @@ plainto_tsquery('simple', $1)
                   OR payload ILIKE '%' || $1 || '%'
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                limit,
            )
            return [(row["session_id"], float(row["score"])) for row in rows]
        finally:
            await self.pool.release(conn)

    async def search_session_ids(self, query: str, *, limit: int = 50) -> list[str]:
        return [session_id for session_id, _score in await self.search(query, limit=limit)]


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PostgresSessionStore",
    "SessionMetadata",
    "create_session_id",
    "DEFAULT_DSN",
]
