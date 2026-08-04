# pi-storage — PostgreSQL 会话存储

对齐 TS [packages/storage/sqlite-node](https://github.com/earendil-works/pi-mono/tree/main/packages/storage)：
Python 侧后端定为 **PostgreSQL**（接口照抄 TS 的 SessionStore / SessionSearch），
搜索用 `tsvector` + `pg_trgm`（TS FTS5 的语义等价实现）。

## 依赖与启动

```bash
# 启动 compose 中的 pg 服务（postgres:16-alpine + 命名卷 + 健康检查）
docker compose -f docker/compose.yaml up -d pg

# 容器内连接用服务名 pg；宿主测试用 127.0.0.1 映射端口
$env:PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
```

## 内容

- `migrations.py` — 顺序迁移（`MIGRATIONS` + `SCHEMA_VERSION` 表）：
  `sessions` / `session_entries` / `session_sequences` / `branch_entries` /
  `session_materialized` / `entry_materialized`（表结构按 TS `001_initial.sql`
  平移 PG 方言）+ `pg_trgm` / `tsvector` 索引。
- `store.py` — `PostgresSessionStore`（asyncpg 池）：
  - 会话：`create_session` / `list_sessions` / `get_session` / `delete_session` /
    `set_leaf_id` / `get_leaf_id`
  - 条目：`append_entry`（事务内分配 `entry_seq`）/ `get_entries` / `get_branch`
  - 搜索：`search` / `search_session_ids`（`ts_rank` + `similarity` 排序）
  - `schema` 参数支持独立 schema 隔离（测试用）

## 用法

```python
import asyncio
from pi_storage import PostgresSessionStore


async def main():
    store = PostgresSessionStore("postgresql://pi:pi@127.0.0.1:5432/pi")
    await store.open()
    await store.migrate()

    meta = await store.create_session("/tmp/proj")
    await store.append_entry(
        meta.id,
        {
            "id": "e1",
            "type": "message",
            "timestamp": "2026-08-04T00:00:00+00:00",
        },
    )
    hits = await store.search_session_ids("hello")
    await store.close()


asyncio.run(main())
```

## 测试

```bash
$env:PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
uv run pytest src/pi_storage/tests/ -v
```

缺 PG 时整模块 skipif 跳过；有 PG 时覆盖增删查、分支、leaf、搜索与错误路径（7 个测试）。

## 许可

MIT
