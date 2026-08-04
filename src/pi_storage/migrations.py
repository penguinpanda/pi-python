"""PG 方言迁移（对齐 TS 001_initial.sql，逐条顺序执行）。"""

from __future__ import annotations

MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        cwd TEXT NOT NULL,
        parent_session_id TEXT NULL,
        metadata TEXT NULL,
        active_leaf_id TEXT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_cwd ON sessions(cwd);
    CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS session_entries (
        session_id TEXT NOT NULL,
        id TEXT NOT NULL,
        entry_seq INTEGER NOT NULL,
        parent_id TEXT NULL,
        type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (session_id, id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_session_entries_session_seq
        ON session_entries(session_id, entry_seq);
    CREATE INDEX IF NOT EXISTS idx_session_entries_session_parent
        ON session_entries(session_id, parent_id);
    CREATE INDEX IF NOT EXISTS idx_session_entries_session_type
        ON session_entries(session_id, type);
    """,
    """
    CREATE TABLE IF NOT EXISTS session_sequences (
        session_id TEXT PRIMARY KEY,
        next_seq INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS branch_entries (
        session_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        entry_seq INTEGER NOT NULL,
        PRIMARY KEY (session_id, branch_id, entry_id)
    );
    CREATE INDEX IF NOT EXISTS idx_branch_entries_session_branch
        ON branch_entries(session_id, branch_id);
    CREATE INDEX IF NOT EXISTS idx_branch_entries_session_branch_seq
        ON branch_entries(session_id, branch_id, entry_seq);
    CREATE INDEX IF NOT EXISTS idx_branch_entries_session_entry
        ON branch_entries(session_id, entry_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS session_materialized (
        session_id TEXT PRIMARY KEY,
        payload TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS entry_materialized (
        session_id TEXT NOT NULL,
        entry_seq INTEGER NOT NULL,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (session_id, entry_seq, type)
    );
    CREATE INDEX IF NOT EXISTS idx_entry_materialized_session_type_seq
        ON entry_materialized(session_id, type, entry_seq);
    """,
    # 搜索：TS FTS5 → PG tsvector + pg_trgm（封装在 repository 内，SQL 不通用）。
    """
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE INDEX IF NOT EXISTS idx_session_entries_payload_trgm
        ON session_entries USING gin (payload gin_trgm_ops);
    CREATE INDEX IF NOT EXISTS idx_session_entries_payload_tsv
        ON session_entries USING gin (to_tsvector('simple', payload));
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


async def apply_migrations(conn) -> int:
    """顺序执行未应用的迁移，返回当前 schema 版本。"""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    current = await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    for index, statement in enumerate(MIGRATIONS, start=1):
        if index <= current:
            continue
        async with conn.transaction():
            await conn.execute(statement)
            await conn.execute(
                "INSERT INTO schema_version (version) VALUES ($1)",
                index,
            )
    return len(MIGRATIONS)


__all__ = ["MIGRATIONS", "SCHEMA_VERSION", "apply_migrations"]
