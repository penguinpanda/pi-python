"""storage — 会话存储（PostgreSQL / Docker，路线图 P2-2）。

接口对齐 TS packages/storage/sqlite-node（SessionStore / SessionSearch），
后端为 PostgreSQL：tsvector + pg_trgm 搜索、顺序迁移文件、schema 版本表。
"""

from .migrations import MIGRATIONS, SCHEMA_VERSION, apply_migrations
from .store import PostgresSessionStore, SessionMetadata
from .v4 import (
    PgSessionSearch,
    PostgresV4SessionRepo,
    PostgresV4SessionStorage,
    create_postgres_v4_repo,
)

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "PostgresSessionStore",
    "PostgresV4SessionRepo",
    "PostgresV4SessionStorage",
    "PgSessionSearch",
    "create_postgres_v4_repo",
    "SessionMetadata",
    "apply_migrations",
]
