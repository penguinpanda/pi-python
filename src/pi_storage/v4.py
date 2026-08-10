"""PostgreSQL v4 会话后端（对齐 TS sqlite-node 语义）。

实现 `pi_agent.session.v4.types.SessionStorage` / `SessionRepo` 协议：
全局 seq、lanes / records / lane_moves / facts、branch cache、session_stats、
writer lease（TTL 30s + 心跳 10s + fence）。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, cast

import asyncpg

from pi_ai.utils.uuid import uuidv7
from pi_agent.session.v4.session import Session
from pi_agent.session.v4.types import (
    BranchEntryQuery,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
    SessionStats,
)

from .migrations import apply_migrations
from .store import DEFAULT_DSN


class PgSessionMetadata(SessionMetadata, total=False):
    """PG 会话元数据（path 为合成标识）。"""

    cwd: str
    path: str
    metadata: dict[str, Any]


class PgSessionCreateOptions(SessionCreateOptions, total=False):
    cwd: str
    metadata: dict[str, Any]


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _ms_from_iso(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _entry_payload(entry: Entry) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if key not in ("type", "id", "seq", "parentId", "timestamp")
    }


def _decode_entry(row: asyncpg.Record) -> Entry:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise SessionError("invalid_entry", f"Invalid payload for entry {row['id']}")
    entry: dict[str, Any] = {
        "id": row["id"],
        "seq": row["entry_seq"],
        "parentId": row["parent_id"],
        "timestamp": _ms_from_iso(row["timestamp"]),
        "type": row["type"],
        **payload,
    }
    return cast(Entry, entry)


def _decode_record(row: asyncpg.Record) -> LaneRecord:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise SessionError("storage", f"Invalid payload for record {row['id']}")
    return cast(
        LaneRecord,
        {
            **payload,
            "seq": row["seq"],
            "timestamp": _ms_from_iso(row["timestamp"]),
        },
    )


# ---------------------------------------------------------------------------
# 底层 SQL 助手
# ---------------------------------------------------------------------------


async def _next_seq(conn: asyncpg.Connection, session_id: str) -> int:
    value = await conn.fetchval(
        """
        UPDATE session_sequences SET next_seq = next_seq + 1
        WHERE session_id = $1 RETURNING next_seq
        """,
        session_id,
    )
    if value is None:
        raise SessionError("storage", f"Missing sequence row for session {session_id}")
    return int(value) - 1


async def _set_seq(conn: asyncpg.Connection, session_id: str, next_seq: int) -> None:
    await conn.execute(
        "UPDATE session_sequences SET next_seq = $1 WHERE session_id = $2",
        next_seq,
        session_id,
    )


async def _insert_entry_row(
    conn: asyncpg.Connection,
    session_id: str,
    *,
    seq: int,
    entry: Entry,
) -> None:
    await conn.execute(
        """
        INSERT INTO session_entries (
            session_id, id, entry_seq, parent_id, type, timestamp, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        session_id,
        entry["id"],
        seq,
        entry["parentId"],
        entry["type"],
        _iso_from_ms(entry["timestamp"]),
        _json_dumps(_entry_payload(entry)),
    )


async def _read_entry_row(
    conn: asyncpg.Connection, session_id: str, entry_id: str
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT session_id, id, entry_seq, parent_id, type, timestamp, payload
        FROM session_entries WHERE session_id = $1 AND id = $2
        """,
        session_id,
        entry_id,
    )


async def _entry_id_exists(conn: asyncpg.Connection, session_id: str, entry_id: str) -> bool:
    return (
        await conn.fetchval(
            "SELECT 1 FROM session_entries WHERE session_id = $1 AND id = $2 LIMIT 1",
            session_id,
            entry_id,
        )
        is not None
    )


async def _record_id_exists(conn: asyncpg.Connection, session_id: str, record_id: str) -> bool:
    return (
        await conn.fetchval(
            "SELECT 1 FROM records WHERE session_id = $1 AND id = $2 LIMIT 1",
            session_id,
            record_id,
        )
        is not None
    )


async def _insert_record_row(
    conn: asyncpg.Connection,
    session_id: str,
    *,
    seq: int,
    record: LaneRecord,
) -> None:
    run_id: str | None = None
    op_kind: str | None = None
    if record["type"] == "operation_started":
        run_id = record["id"]
        op_kind = cast(dict[str, Any], record["intent"])["kind"]
    elif "runId" in cast(dict[str, Any], record):
        run_id = cast(dict[str, Any], record)["runId"]
    await conn.execute(
        """
        INSERT INTO records (
            session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        session_id,
        seq,
        record["id"],
        record["lane"],
        run_id,
        record["type"],
        op_kind,
        _iso_from_ms(record["timestamp"]),
        _json_dumps({k: v for k, v in record.items() if k not in ("seq", "timestamp")}),
    )


# ---------------------------------------------------------------------------
# lanes / facts / stats / lease / branch cache
# ---------------------------------------------------------------------------


async def _read_lanes(conn: asyncpg.Connection, session_id: str) -> list[LanePointer]:
    rows = await conn.fetch(
        """
        SELECT l.lane, l.leaf_id,
               (l.leaf_id IS NULL OR EXISTS (
                   SELECT 1 FROM session_entries AS e
                   WHERE e.session_id = l.session_id AND e.id = l.leaf_id
               )) AS leaf_exists
        FROM lanes AS l WHERE l.session_id = $1 ORDER BY l.lane
        """,
        session_id,
    )
    result: list[LanePointer] = []
    for row in rows:
        if not bool(row["leaf_exists"]):
            raise SessionError(
                "storage",
                f"Lane {row['lane']} points at missing entry {row['leaf_id']}",
            )
        result.append({"lane": row["lane"], "leafId": row["leaf_id"]})
    return result


async def _read_lane_leaf(conn: asyncpg.Connection, session_id: str, lane: str) -> str | None:
    row = await conn.fetchrow(
        """
        SELECT l.leaf_id,
               (l.leaf_id IS NULL OR EXISTS (
                   SELECT 1 FROM session_entries AS e
                   WHERE e.session_id = l.session_id AND e.id = l.leaf_id
               )) AS leaf_exists
        FROM lanes AS l WHERE l.session_id = $1 AND l.lane = $2
        """,
        session_id,
        lane,
    )
    if row is None:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    if not bool(row["leaf_exists"]):
        raise SessionError("storage", f"Entry {row['leaf_id']} not found")
    return row["leaf_id"]


async def _lane_exists(conn: asyncpg.Connection, session_id: str, lane: str) -> bool:
    return (
        await conn.fetchval(
            "SELECT 1 FROM lanes WHERE session_id = $1 AND lane = $2 LIMIT 1",
            session_id,
            lane,
        )
        is not None
    )


async def _insert_lane(
    conn: asyncpg.Connection,
    session_id: str,
    seq: int,
    lane: str,
    leaf_id: str | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO lanes (session_id, lane, leaf_id, open_operation_id)
        VALUES ($1, $2, $3, NULL)
        """,
        session_id,
        lane,
        leaf_id,
    )
    await conn.execute(
        """
        INSERT INTO lane_moves (session_id, seq, lane, leaf_id)
        VALUES ($1, $2, $3, $4)
        """,
        session_id,
        seq,
        lane,
        leaf_id,
    )


async def _move_lane(
    conn: asyncpg.Connection,
    session_id: str,
    seq: int,
    lane: str,
    leaf_id: str | None,
) -> None:
    result = await conn.execute(
        "UPDATE lanes SET leaf_id = $1 WHERE session_id = $2 AND lane = $3",
        leaf_id,
        session_id,
        lane,
    )
    if result != "UPDATE 1":
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    await conn.execute(
        """
        INSERT INTO lane_moves (session_id, seq, lane, leaf_id)
        VALUES ($1, $2, $3, $4)
        """,
        session_id,
        seq,
        lane,
        leaf_id,
    )


async def _set_lane_leaf(
    conn: asyncpg.Connection, session_id: str, lane: str, leaf_id: str | None
) -> None:
    await conn.execute(
        "UPDATE lanes SET leaf_id = $1 WHERE session_id = $2 AND lane = $3",
        leaf_id,
        session_id,
        lane,
    )


async def _start_lane_operation(
    conn: asyncpg.Connection, session_id: str, lane: str, run_id: str
) -> None:
    result = await conn.execute(
        """
        UPDATE lanes SET open_operation_id = $1
        WHERE session_id = $2 AND lane = $3 AND open_operation_id IS NULL
        """,
        run_id,
        session_id,
        lane,
    )
    if result == "UPDATE 1":
        return
    row = await conn.fetchrow(
        "SELECT open_operation_id FROM lanes WHERE session_id = $1 AND lane = $2",
        session_id,
        lane,
    )
    if row is None:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    raise SessionError(
        "storage",
        f"Lane {lane} already has an open operation {row['open_operation_id']}",
    )


async def _finish_lane_operation(
    conn: asyncpg.Connection, session_id: str, lane: str, run_id: str
) -> None:
    await conn.execute(
        """
        UPDATE lanes SET open_operation_id = NULL
        WHERE session_id = $1 AND lane = $2 AND open_operation_id = $3
        """,
        session_id,
        lane,
        run_id,
    )


async def _append_fact(
    conn: asyncpg.Connection,
    session_id: str,
    seq: int,
    kind: str,
    key: str | None,
    value: str | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO facts (session_id, seq, kind, key, value)
        VALUES ($1, $2, $3, $4, $5)
        """,
        session_id,
        seq,
        kind,
        key,
        value,
    )


async def _read_latest_fact(
    conn: asyncpg.Connection, session_id: str, kind: str, key: str | None
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT session_id, seq, kind, key, value FROM facts
        WHERE session_id = $1 AND kind = $2 AND key IS NOT DISTINCT FROM $3
        ORDER BY seq DESC LIMIT 1
        """,
        session_id,
        kind,
        key,
    )


async def _read_latest_label_facts(
    conn: asyncpg.Connection, session_id: str
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT key, value FROM (
            SELECT key, value, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rank
            FROM facts WHERE session_id = $1 AND kind = 'label'
        ) ranked WHERE rank = 1 AND value IS NOT NULL ORDER BY key
        """,
        session_id,
    )


async def _read_stats(conn: asyncpg.Connection, session_id: str) -> SessionStats:
    row = await conn.fetchrow(
        """
        SELECT message_count, cached_tokens, uncached_tokens, total_tokens, cost_total
        FROM session_stats WHERE session_id = $1
        """,
        session_id,
    )
    if row is None:
        raise SessionError("storage", f"Missing stats row for session {session_id}")
    return {
        "messageCount": int(row["message_count"]),
        "cachedTokens": int(row["cached_tokens"]),
        "uncachedTokens": int(row["uncached_tokens"]),
        "totalTokens": int(row["total_tokens"]),
        "costTotal": float(row["cost_total"]),
    }


async def _increment_message_count(conn: asyncpg.Connection, session_id: str) -> None:
    result = await conn.execute(
        "UPDATE session_stats SET message_count = message_count + 1 WHERE session_id = $1",
        session_id,
    )
    if result != "UPDATE 1":
        raise SessionError("storage", f"Missing stats row for session {session_id}")


async def _add_usage_to_stats(
    conn: asyncpg.Connection, session_id: str, usage: dict[str, Any]
) -> None:
    cost = usage.get("cost") or {}
    result = await conn.execute(
        """
        UPDATE session_stats
        SET cached_tokens = cached_tokens + $1,
            uncached_tokens = uncached_tokens + $2,
            total_tokens = total_tokens + $3,
            cost_total = cost_total + $4
        WHERE session_id = $5
        """,
        usage.get("cache_read", 0),
        usage.get("input", 0) + usage.get("cache_write", 0),
        usage.get("total_tokens", 0),
        cost.get("total", 0),
        session_id,
    )
    if result != "UPDATE 1":
        raise SessionError("storage", f"Missing stats row for session {session_id}")


# ---------------------------------------------------------------------------
# writer lease
# ---------------------------------------------------------------------------


class WriterLease:
    def __init__(self, owner_id: str, fence: int, expires_at_ms: int) -> None:
        self.owner_id = owner_id
        self.fence = fence
        self.expires_at_ms = expires_at_ms


async def _acquire_lease(
    conn: asyncpg.Connection, session_id: str, owner_id: str
) -> WriterLease | None:
    now = _now_ms()
    row = await conn.fetchrow(
        """
        INSERT INTO writer_leases (session_id, owner_id, fence, expires_at_ms)
        VALUES ($1, $2, 1, $3)
        ON CONFLICT (session_id) DO UPDATE SET
            owner_id = EXCLUDED.owner_id,
            fence = writer_leases.fence + 1,
            expires_at_ms = EXCLUDED.expires_at_ms
        WHERE writer_leases.expires_at_ms <= $4
        RETURNING owner_id, fence, expires_at_ms
        """,
        session_id,
        owner_id,
        now + 30_000,
        now,
    )
    if row is None:
        return None
    return WriterLease(row["owner_id"], int(row["fence"]), int(row["expires_at_ms"]))


async def _renew_lease(conn: asyncpg.Connection, session_id: str, lease: WriterLease) -> bool:
    now = _now_ms()
    result = await conn.execute(
        """
        UPDATE writer_leases SET expires_at_ms = $1
        WHERE session_id = $2 AND owner_id = $3 AND fence = $4 AND expires_at_ms > $5
        """,
        now + 30_000,
        session_id,
        lease.owner_id,
        lease.fence,
        now,
    )
    if result == "UPDATE 1":
        lease.expires_at_ms = now + 30_000
        return True
    return False


async def _release_lease(conn: asyncpg.Connection, session_id: str, lease: WriterLease) -> None:
    await conn.execute(
        """
        DELETE FROM writer_leases WHERE session_id = $1 AND owner_id = $2 AND fence = $3
        """,
        session_id,
        lease.owner_id,
        lease.fence,
    )


# ---------------------------------------------------------------------------
# branch cache
# ---------------------------------------------------------------------------


async def _insert_branch_entry(
    conn: asyncpg.Connection,
    session_id: str,
    branch_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO branch_entries (
            session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        session_id,
        branch_id,
        entry_id,
        entry_seq,
        entry_type,
        custom_type,
    )


async def _insert_branch_tip(
    conn: asyncpg.Connection, session_id: str, tip_id: str, branch_id: str
) -> None:
    await conn.execute(
        """
        INSERT INTO branch_tips (session_id, tip_id, branch_id)
        VALUES ($1, $2, $3)
        """,
        session_id,
        tip_id,
        branch_id,
    )


async def _read_tip_branch_id(conn: asyncpg.Connection, session_id: str, tip_id: str) -> str | None:
    return await conn.fetchval(
        "SELECT branch_id FROM branch_tips WHERE session_id = $1 AND tip_id = $2",
        session_id,
        tip_id,
    )


async def _update_tip(
    conn: asyncpg.Connection,
    session_id: str,
    branch_id: str,
    old_tip_id: str,
    new_tip_id: str,
) -> bool:
    result = await conn.execute(
        """
        UPDATE branch_tips SET tip_id = $1
        WHERE session_id = $2 AND branch_id = $3 AND tip_id = $4
        """,
        new_tip_id,
        session_id,
        branch_id,
        old_tip_id,
    )
    return result == "UPDATE 1"


async def _read_branch_containing_entry(
    conn: asyncpg.Connection, session_id: str, entry_id: str
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT branch_id, entry_seq FROM branch_entries
        WHERE session_id = $1 AND entry_id = $2
        ORDER BY branch_id LIMIT 1
        """,
        session_id,
        entry_id,
    )


async def _copy_branch_through_seq(
    conn: asyncpg.Connection,
    session_id: str,
    target_branch_id: str,
    source_branch_id: str,
    through_seq: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO branch_entries (
            session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
        )
        SELECT session_id, $1, entry_id, entry_seq, entry_type, custom_type
        FROM branch_entries
        WHERE session_id = $2 AND branch_id = $3 AND entry_seq <= $4
        """,
        target_branch_id,
        session_id,
        source_branch_id,
        through_seq,
    )


async def _build_cached_branch(conn: asyncpg.Connection, session_id: str, leaf_id: str) -> None:
    branch_id = uuidv7()
    await conn.execute(
        """
        WITH RECURSIVE path(id, entry_seq, parent_id, type, custom_type) AS (
            SELECT id, entry_seq, parent_id, type,
                   CASE WHEN type = 'custom' THEN payload::jsonb->>'customType' ELSE NULL END
            FROM session_entries WHERE session_id = $1 AND id = $2
            UNION ALL
            SELECT parent.id, parent.entry_seq, parent.parent_id, parent.type,
                   CASE WHEN parent.type = 'custom' THEN parent.payload::jsonb->>'customType' ELSE NULL END
            FROM session_entries AS parent
            JOIN path AS child ON child.parent_id = parent.id
            WHERE parent.session_id = $1
        )
        INSERT INTO branch_entries (
            session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
        )
        SELECT $1, $3, id, entry_seq, type, custom_type FROM path
        """,
        session_id,
        leaf_id,
        branch_id,
    )
    await _insert_branch_tip(conn, session_id, leaf_id, branch_id)


async def _delete_branch_cache(conn: asyncpg.Connection, session_id: str) -> None:
    await conn.execute("DELETE FROM branch_tips WHERE session_id = $1", session_id)
    await conn.execute("DELETE FROM branch_entries WHERE session_id = $1", session_id)


async def _rebuild_branch_cache(conn: asyncpg.Connection, session_id: str) -> None:
    tips = await conn.fetch(
        """
        SELECT leaf.id FROM session_entries AS leaf
        WHERE leaf.session_id = $1 AND NOT EXISTS (
            SELECT 1 FROM session_entries AS child
            WHERE child.session_id = leaf.session_id AND child.parent_id = leaf.id
        )
        ORDER BY leaf.entry_seq
        """,
        session_id,
    )
    await _delete_branch_cache(conn, session_id)
    for tip in tips:
        await _build_cached_branch(conn, session_id, tip["id"])


async def _append_entry_to_branch_cache(
    conn: asyncpg.Connection,
    session_id: str,
    entry: Entry,
) -> None:
    custom_type: str | None = (
        cast(dict[str, Any], entry).get("customType") if entry["type"] == "custom" else None
    )
    parent_id = entry["parentId"]
    if parent_id is None:
        branch_id = uuidv7()
        await _insert_branch_entry(
            conn, session_id, branch_id, entry["id"], entry["seq"], entry["type"], custom_type
        )
        await _insert_branch_tip(conn, session_id, entry["id"], branch_id)
        return
    tip_branch_id = await _read_tip_branch_id(conn, session_id, parent_id)
    if tip_branch_id is not None:
        await _insert_branch_entry(
            conn, session_id, tip_branch_id, entry["id"], entry["seq"], entry["type"], custom_type
        )
        if not await _update_tip(conn, session_id, tip_branch_id, parent_id, entry["id"]):
            raise SessionError("invalid_entry", f"Branch tip {parent_id} changed during append")
        return
    source = await _read_branch_containing_entry(conn, session_id, parent_id)
    if source is None:
        raise SessionError(
            "invalid_entry", f"Branch cache has no branch containing parent entry {parent_id}"
        )
    branch_id = uuidv7()
    await _copy_branch_through_seq(
        conn, session_id, branch_id, source["branch_id"], int(source["entry_seq"])
    )
    await _insert_branch_entry(
        conn, session_id, branch_id, entry["id"], entry["seq"], entry["type"], custom_type
    )
    await _insert_branch_tip(conn, session_id, entry["id"], branch_id)


async def _read_cached_branch(
    conn: asyncpg.Connection, session_id: str, leaf_id: str
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT branch_id, entry_seq FROM branch_entries
        WHERE session_id = $1 AND entry_id = $2
        ORDER BY branch_id LIMIT 1
        """,
        session_id,
        leaf_id,
    )


async def _query_cached_branch_rows(
    conn: asyncpg.Connection,
    session_id: str,
    branch: asyncpg.Record,
    query: dict[str, Any],
) -> list[asyncpg.Record]:
    oldest_first = query.get("order") == "oldestFirst"
    branch_id = branch["branch_id"]
    leaf_seq = int(branch["entry_seq"])
    stop_type = query.get("stopAtType")
    stop_id = query.get("stopAtId")
    stop_predicates: list[str] = []
    boundary_params: list[Any] = [session_id, branch_id, leaf_seq]
    if stop_type is not None:
        stop_predicates.append("stop_entry.type = $%d" % (len(boundary_params) + 1))
        boundary_params.append(stop_type)
    if stop_id is not None:
        stop_predicates.append("stop.entry_id = $%d" % (len(boundary_params) + 1))
        boundary_params.append(stop_id)
    boundary_sql = ""
    range_sql = ""
    use_boundary = bool(stop_predicates)
    if use_boundary:
        boundary_sql = (
            "WITH boundary AS ("
            f"SELECT {'MIN' if oldest_first else 'MAX'}(stop.entry_seq) AS entry_seq "
            "FROM branch_entries AS stop "
            "JOIN session_entries AS stop_entry "
            "ON stop_entry.session_id = stop.session_id AND stop_entry.id = stop.entry_id "
            "WHERE stop.session_id = $1 AND stop.branch_id = $2 AND stop.entry_seq <= $3 "
            f"AND ({' OR '.join(stop_predicates)})"
            ") "
        )
        range_sql = (
            " AND b.entry_seq "
            f"{'<=' if oldest_first else '>='} COALESCE("
            f"(SELECT entry_seq FROM boundary), {leaf_seq if oldest_first else 0})"
        )
    sql = (
        f"{boundary_sql}"
        "SELECT e.id, e.entry_seq, e.parent_id, e.type, e.timestamp, e.payload "
        "FROM branch_entries AS b "
        "JOIN session_entries AS e ON e.session_id = b.session_id AND e.id = b.entry_id "
        "WHERE b.session_id = $1 AND b.branch_id = $2 AND b.entry_seq <= $3"
        f"{range_sql} "
        f"ORDER BY b.entry_seq {'ASC' if oldest_first else 'DESC'}"
    )
    params = boundary_params if use_boundary else [session_id, branch_id, leaf_seq]
    return await conn.fetch(sql, *params)


def _validate_cached_branch_rows(rows: list[asyncpg.Record], query: dict[str, Any]) -> None:
    if not rows:
        return
    path = sorted(rows, key=lambda row: int(row["entry_seq"]))
    if query.get("stopAtId") is None and query.get("stopAtType") is None:
        if path[0]["parent_id"] is not None:
            raise SessionError("invalid_entry", f"Entry {path[0]['parent_id']} not found")
    for index in range(1, len(path)):
        previous = path[index - 1]
        current = path[index]
        if current["parent_id"] != previous["id"]:
            raise SessionError("invalid_entry", f"Entry {current['parent_id']} not found")


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


class PostgresV4SessionStorage:
    """v4 SessionStorage 的 PostgreSQL 实现（一个会话一个 lease）。"""

    def __init__(
        self,
        repo: "PostgresV4SessionRepo",
        metadata: PgSessionMetadata,
        lease: WriterLease,
    ) -> None:
        self._repo = repo
        self._metadata: PgSessionMetadata = cast(PgSessionMetadata, dict(metadata))
        self._lease = lease
        self._lock = asyncio.Lock()
        self._closed = False
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    @property
    def session_id(self) -> str:
        return self._metadata["id"]

    async def _write(self, fn):
        async with self._lock:
            if self._closed:
                raise SessionError("storage", f"Session {self.session_id} is closed")
            conn = await self._repo._acquire()
            try:
                async with conn.transaction():
                    if not await _renew_lease(conn, self.session_id, self._lease):
                        raise SessionError(
                            "storage", f"Session {self.session_id} writer lease was lost"
                        )
                    return await fn(conn)
            finally:
                await self._repo.pool.release(conn)

    async def _heartbeat(self) -> None:
        while not self._closed:
            await asyncio.sleep(10)
            if self._closed:
                return
            conn = await self._repo._acquire()
            try:
                await _renew_lease(conn, self.session_id, self._lease)
            except Exception:
                pass
            finally:
                await self._repo.pool.release(conn)

    async def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_task.cancel()
        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass
        conn = await self._repo._acquire()
        try:
            await _release_lease(conn, self.session_id, self._lease)
        finally:
            await self._repo.pool.release(conn)

    async def get_metadata(self) -> SessionMetadata:
        return cast(SessionMetadata, dict(self._metadata))

    async def get_lanes(self) -> list[LanePointer]:
        conn = await self._repo._acquire()
        try:
            return await _read_lanes(conn, self.session_id)
        finally:
            await self._repo.pool.release(conn)

    async def create_lane(self, lane: str, at: str | None) -> None:
        async def _run(conn: asyncpg.Connection) -> None:
            if await _lane_exists(conn, self.session_id, lane):
                raise SessionError("already_exists", f"Lane already exists: {lane}")
            if at is not None and not await _entry_id_exists(conn, self.session_id, at):
                raise SessionError("not_found", f"Entry not found: {at}")
            seq = await _next_seq(conn, self.session_id)
            await _insert_lane(conn, self.session_id, seq, lane, at)

        await self._write(_run)

    async def move_lane(self, lane: str, to: str | None) -> None:
        async def _run(conn: asyncpg.Connection) -> None:
            if not await _lane_exists(conn, self.session_id, lane):
                raise SessionError("invalid_lane", f"Lane not found: {lane}")
            if to is not None and not await _entry_id_exists(conn, self.session_id, to):
                raise SessionError("not_found", f"Entry not found: {to}")
            seq = await _next_seq(conn, self.session_id)
            await _move_lane(conn, self.session_id, seq, lane, to)

        await self._write(_run)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        async def _run(conn: asyncpg.Connection) -> Entry:
            parent_id = await _read_lane_leaf(conn, self.session_id, lane)
            if await _entry_id_exists(
                conn, self.session_id, entry["id"]
            ) or await _record_id_exists(conn, self.session_id, entry["id"]):
                raise SessionError("already_exists", f"Session id already exists: {entry['id']}")
            seq = await _next_seq(conn, self.session_id)
            committed = cast(
                Entry,
                {
                    **{k: v for k, v in entry.items()},
                    "parentId": parent_id,
                    "seq": seq,
                    "timestamp": _now_ms(),
                },
            )
            await _insert_entry_row(conn, self.session_id, seq=seq, entry=committed)
            await _set_lane_leaf(conn, self.session_id, lane, committed["id"])
            await _append_entry_to_branch_cache(conn, self.session_id, committed)
            if committed["type"] == "message":
                await _increment_message_count(conn, self.session_id)
            return committed

        return await self._write(_run)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        async def _run(conn: asyncpg.Connection) -> LaneRecord:
            if not await _lane_exists(conn, self.session_id, record["lane"]):
                raise SessionError("invalid_lane", f"Lane not found: {record['lane']}")
            if await _entry_id_exists(
                conn, self.session_id, record["id"]
            ) or await _record_id_exists(conn, self.session_id, record["id"]):
                raise SessionError("already_exists", f"Session id already exists: {record['id']}")
            seq = await _next_seq(conn, self.session_id)
            committed = cast(
                LaneRecord,
                {**record, "seq": seq, "timestamp": _now_ms()},
            )
            if committed["type"] == "operation_started":
                await _start_lane_operation(
                    conn, self.session_id, committed["lane"], committed["id"]
                )
            await _insert_record_row(conn, self.session_id, seq=seq, record=committed)
            if committed["type"] == "operation_finished":
                await _finish_lane_operation(
                    conn,
                    self.session_id,
                    committed["lane"],
                    cast(dict[str, Any], committed)["runId"],
                )
            if committed["type"] == "usage":
                await _add_usage_to_stats(
                    conn, self.session_id, cast(dict[str, Any], committed)["usage"]
                )
            return committed

        return await self._write(_run)

    async def get_entry(self, entry_id: str) -> Entry | None:
        conn = await self._repo._acquire()
        try:
            row = await _read_entry_row(conn, self.session_id, entry_id)
            return _decode_entry(row) if row is not None else None
        finally:
            await self._repo.pool.release(conn)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        query = query or {}
        order = "ASC" if query.get("order") == "oldestFirst" else "DESC"
        conn = await self._repo._acquire()
        try:
            rows = await conn.fetch(
                f"""
                SELECT id, entry_seq, parent_id, type, timestamp, payload
                FROM session_entries WHERE session_id = $1 ORDER BY entry_seq {order}
                """,
                self.session_id,
            )
            entries = [_decode_entry(row) for row in rows]
            return _filter_entries(entries, query)
        finally:
            await self._repo.pool.release(conn)

    async def find_entries_on_branch(self, query: BranchEntryQuery | None = None) -> list[Entry]:
        branch_query: dict[str, Any] = dict(query or {})
        start = branch_query.get("start")
        if not isinstance(start, str):
            raise SessionError("invalid_query", "branch query requires start")
        conn = await self._repo._acquire()
        try:
            branch = await _read_cached_branch(conn, self.session_id, start)
            if branch is None:
                if await _read_entry_row(conn, self.session_id, start) is None:
                    raise SessionError("not_found", f"Entry not found: {start}")
                raise SessionError("invalid_entry", f"Branch cache missing entry {start}")
            rows = await _query_cached_branch_rows(conn, self.session_id, branch, branch_query)
            _validate_cached_branch_rows(rows, branch_query)
            entries = [_decode_entry(row) for row in rows]
            return _filter_entries(entries, cast(EntryQuery, branch_query))
        finally:
            await self._repo.pool.release(conn)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        query = query or {}
        predicates = ["session_id = $1"]
        params: list[Any] = [self.session_id]
        if query.get("lane") is not None:
            predicates.append("lane = $%d" % (len(params) + 1))
            params.append(query["lane"])
        if query.get("type") is not None:
            predicates.append("type = $%d" % (len(params) + 1))
            params.append(query["type"])
        if query.get("runId") is not None:
            predicates.append("run_id = $%d" % (len(params) + 1))
            params.append(query["runId"])
        if query.get("operationKind") is not None:
            predicates.append("op_kind = $%d" % (len(params) + 1))
            params.append(query["operationKind"])
        if query.get("afterSeq") is not None:
            predicates.append("seq > $%d" % (len(params) + 1))
            params.append(query["afterSeq"])
        order = "ASC" if query.get("order") == "oldestFirst" else "DESC"
        limit_sql = ""
        if query.get("limit") is not None:
            limit_sql = " LIMIT $%d" % (len(params) + 1)
            params.append(query["limit"])
        conn = await self._repo._acquire()
        try:
            rows = await conn.fetch(
                f"""
                SELECT seq, id, lane, run_id, type, op_kind, timestamp, payload
                FROM records WHERE {" AND ".join(predicates)}
                ORDER BY seq {order}{limit_sql}
                """,
                *params,
            )
            return [_decode_record(row) for row in rows]
        finally:
            await self._repo.pool.release(conn)

    async def find_open_operations(
        self, lane: str, options: dict[str, int] | None = None
    ) -> list[OperationStartedRecord]:
        del options
        conn = await self._repo._acquire()
        try:
            row = await conn.fetchrow(
                "SELECT open_operation_id FROM lanes WHERE session_id = $1 AND lane = $2",
                self.session_id,
                lane,
            )
            if row is None or row["open_operation_id"] is None:
                return []
            record_row = await conn.fetchrow(
                """
                SELECT seq, id, lane, run_id, type, op_kind, timestamp, payload
                FROM records WHERE session_id = $1 AND id = $2
                """,
                self.session_id,
                row["open_operation_id"],
            )
            if record_row is None:
                raise SessionError(
                    "storage",
                    f"Lane {lane} points at missing open operation {row['open_operation_id']}",
                )
            record = _decode_record(record_row)
            if record["type"] != "operation_started":
                raise SessionError("storage", f"Lane {lane} points at invalid open operation")
            return [cast(OperationStartedRecord, record)]
        finally:
            await self._repo.pool.release(conn)

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options or {}
        after_seq = options.get("afterSeq")
        conn = await self._repo._acquire()
        try:
            entry_rows = await conn.fetch(
                """
                SELECT id, entry_seq AS seq, parent_id, type, timestamp, payload
                FROM session_entries WHERE session_id = $1 AND entry_seq > $2
                ORDER BY entry_seq
                """,
                self.session_id,
                after_seq or 0,
            )
            record_rows = await conn.fetch(
                """
                SELECT seq, id, lane, run_id, type, op_kind, timestamp, payload
                FROM records WHERE session_id = $1 AND seq > $2 ORDER BY seq
                """,
                self.session_id,
                after_seq or 0,
            )
            lane_rows = await conn.fetch(
                """
                SELECT seq, lane, leaf_id FROM lane_moves
                WHERE session_id = $1 AND seq > $2 ORDER BY seq
                """,
                self.session_id,
                after_seq or 0,
            )
            fact_rows = await conn.fetch(
                """
                SELECT seq, kind, key, value FROM facts
                WHERE session_id = $1 AND seq > $2 ORDER BY seq
                """,
                self.session_id,
                after_seq or 0,
            )
        finally:
            await self._repo.pool.release(conn)
        log: list[LogItem] = []
        for row in entry_rows:
            log.append({"kind": "entry", "seq": row["seq"], "entry": _decode_entry(row)})
        for row in record_rows:
            log.append({"kind": "record", "seq": row["seq"], "record": _decode_record(row)})
        for row in lane_rows:
            log.append(
                {
                    "kind": "lane",
                    "seq": row["seq"],
                    "lane": row["lane"],
                    "leafId": row["leaf_id"],
                }
            )
        for row in fact_rows:
            if row["kind"] == "name":
                log.append(
                    {
                        "kind": "fact",
                        "seq": row["seq"],
                        "fact": "name",
                        "name": json.loads(row["value"] or "null"),
                    }
                )
            else:
                log.append(
                    {
                        "kind": "fact",
                        "seq": row["seq"],
                        "fact": "label",
                        "targetId": row["key"] or "",
                        "label": None if row["value"] is None else json.loads(row["value"]),
                    }
                )
        log.sort(key=lambda item: item["seq"])
        limit = options.get("limit")
        return log if limit is None else log[:limit]

    async def get_name(self) -> str | None:
        conn = await self._repo._acquire()
        try:
            row = await _read_latest_fact(conn, self.session_id, "name", None)
            if row is None or row["value"] is None:
                return None
            value = json.loads(row["value"])
            return value if isinstance(value, str) else None
        finally:
            await self._repo.pool.release(conn)

    async def set_name(self, name: str) -> None:
        async def _run(conn: asyncpg.Connection) -> None:
            seq = await _next_seq(conn, self.session_id)
            await _append_fact(conn, self.session_id, seq, "name", None, _json_dumps(name))

        await self._write(_run)

    async def get_label(self, entry_id: str) -> str | None:
        conn = await self._repo._acquire()
        try:
            row = await _read_latest_fact(conn, self.session_id, "label", entry_id)
            if row is None or row["value"] is None:
                return None
            value = json.loads(row["value"])
            return value if isinstance(value, str) else None
        finally:
            await self._repo.pool.release(conn)

    async def set_label(self, entry_id: str, label: str | None) -> None:
        async def _run(conn: asyncpg.Connection) -> None:
            if not await _entry_id_exists(conn, self.session_id, entry_id):
                raise SessionError("not_found", f"Entry not found: {entry_id}")
            seq = await _next_seq(conn, self.session_id)
            await _append_fact(
                conn,
                self.session_id,
                seq,
                "label",
                entry_id,
                None if label is None else _json_dumps(label),
            )

        await self._write(_run)

    async def get_stats(self) -> SessionStats:
        conn = await self._repo._acquire()
        try:
            return await _read_stats(conn, self.session_id)
        finally:
            await self._repo.pool.release(conn)


def _filter_entries(entries: list[Entry], query: EntryQuery) -> list[Entry]:
    results: list[Entry] = []
    for entry in entries:
        if query.get("type") is not None and entry["type"] != query["type"]:
            continue
        if query.get("customType") is not None and not (
            entry["type"] == "custom"
            and cast(dict[str, Any], entry).get("customType") == query["customType"]
        ):
            continue
        cursor = query.get("cursor")
        if cursor is not None:
            after_seq = cursor["afterSeq"]
            if query.get("order") == "oldestFirst":
                if entry["seq"] <= after_seq:
                    continue
            elif entry["seq"] >= after_seq:
                continue
        results.append(entry)
        if query.get("limit") is not None and len(results) >= query["limit"]:
            break
    return results


def _decode_session_row(row: Any, path: str) -> PgSessionMetadata:
    metadata: PgSessionMetadata = {
        "id": row["id"],
        "createdAt": _ms_from_iso(row["created_at"]),
        "cwd": row["cwd"],
        "path": path,
    }
    if row.get("parent_session_id") is not None:
        metadata["parentSessionId"] = row["parent_session_id"]
    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            metadata["metadata"] = json.loads(raw_metadata)
        except (ValueError, TypeError):
            pass
    if bool(row.get("has_session_name", False)):
        metadata["metadata"] = metadata.get("metadata") or {}
    return metadata


class PostgresV4SessionRepo:
    """v4 SessionRepo 的 PostgreSQL 实现（repo 自有连接池 + writer lease）。"""

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
        self._active_storages: dict[str, PostgresV4SessionStorage] = {}
        self._ops_lock = asyncio.Lock()

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Repo is not open; call open() first")
        return self._pool

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._min_size, max_size=self._max_size
        )
        if self._schema:
            await self._pool.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
        await self.migrate()

    async def migrate(self) -> int:
        conn = await self._acquire()
        try:
            return await apply_migrations(conn)
        finally:
            await self.pool.release(conn)

    async def close(self) -> None:
        for storage in list(self._active_storages.values()):
            await storage.release()
        self._active_storages.clear()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _acquire(self) -> asyncpg.Connection:
        conn = await self.pool.acquire()
        if self._schema:
            await conn.execute(f'SET search_path TO "{self._schema}", public')
        return conn

    def _path(self, session_id: str) -> str:
        return f"pg:{self._schema or 'public'}:{session_id}"

    async def _require_session(self, conn: asyncpg.Connection, session_id: str) -> asyncpg.Record:
        row = await conn.fetchrow(
            """
            SELECT s.id, s.created_at, s.cwd, s.parent_session_id, s.metadata,
                   name_fact.value IS NOT NULL AS has_session_name
            FROM sessions AS s
            LEFT JOIN facts AS name_fact
                ON name_fact.session_id = s.id AND name_fact.kind = 'name'
                AND name_fact.key IS NULL
                AND name_fact.seq = (
                    SELECT MAX(f.seq) FROM facts AS f
                    WHERE f.session_id = s.id AND f.kind = 'name' AND f.key IS NULL
                )
            WHERE s.id = $1
            """,
            session_id,
        )
        if row is None:
            raise SessionError("not_found", f"Session not found: {session_id}")
        return row

    def _storage_from_lease(self, metadata: PgSessionMetadata, lease: WriterLease) -> Session:
        storage = PostgresV4SessionStorage(self, metadata, lease)
        self._active_storages[metadata["id"]] = storage
        return Session(storage)

    async def _claim_with_conn(
        self, conn: asyncpg.Connection, metadata: PgSessionMetadata
    ) -> Session:
        active = self._active_storages.get(metadata["id"])
        if active is not None:
            return Session(active)
        await self._require_session(conn, metadata["id"])
        lease = await _acquire_lease(conn, metadata["id"], uuidv7())
        if lease is None:
            raise SessionError(
                "storage", f"PG session {metadata['id']} already has an active writer"
            )
        return self._storage_from_lease(metadata, lease)

    async def create(self, options: PgSessionCreateOptions | None = None) -> Session:
        options = options or {}
        session_id = options.get("id") or uuidv7()
        async with self._ops_lock:
            conn = await self._acquire()
            try:
                async with conn.transaction():
                    existing = await conn.fetchval(
                        "SELECT 1 FROM sessions WHERE id = $1", session_id
                    )
                    if existing is not None:
                        raise SessionError(
                            "already_exists", f"Session already exists: {session_id}"
                        )
                    created_at = _now_ms()
                    await conn.execute(
                        """
                        INSERT INTO sessions (id, created_at, cwd, parent_session_id, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        session_id,
                        _iso_from_ms(created_at),
                        options.get("cwd", "."),
                        options.get("parentSessionId"),
                        _json_dumps(options["metadata"])
                        if options.get("metadata") is not None
                        else None,
                    )
                    await conn.execute(
                        "INSERT INTO session_sequences (session_id, next_seq) VALUES ($1, 1)",
                        session_id,
                    )
                    await conn.execute(
                        """
                        INSERT INTO session_stats (
                            session_id, message_count, cached_tokens, uncached_tokens,
                            total_tokens, cost_total
                        ) VALUES ($1, 0, 0, 0, 0, 0)
                        """,
                        session_id,
                    )
                    await conn.execute(
                        """
                        INSERT INTO lanes (session_id, lane, leaf_id, open_operation_id)
                        VALUES ($1, 'main', NULL, NULL)
                        """,
                        session_id,
                    )
                    metadata: PgSessionMetadata = {
                        "id": session_id,
                        "createdAt": created_at,
                        "cwd": options.get("cwd", "."),
                        "path": self._path(session_id),
                    }
                    if options.get("parentSessionId") is not None:
                        metadata["parentSessionId"] = options["parentSessionId"]
                    if options.get("metadata") is not None:
                        metadata["metadata"] = options["metadata"]
                    session = await self._claim_with_conn(conn, metadata)
                    return session
            finally:
                await self.pool.release(conn)

    async def open(self, metadata: SessionMetadata) -> Session:
        pg_metadata = cast(PgSessionMetadata, metadata)
        async with self._ops_lock:
            conn = await self._acquire()
            try:
                async with conn.transaction():
                    return await self._claim_with_conn(conn, pg_metadata)
            finally:
                await self.pool.release(conn)

    async def list(self, options: dict[str, Any] | None = None) -> list[PgSessionMetadata]:
        options = options or {}
        conn = await self._acquire()
        try:
            if options.get("cwd") is not None:
                rows = await conn.fetch(
                    """
                    SELECT s.id, s.created_at, s.cwd, s.parent_session_id, s.metadata,
                           name_fact.value IS NOT NULL AS has_session_name
                    FROM sessions AS s
                    LEFT JOIN facts AS name_fact
                        ON name_fact.session_id = s.id AND name_fact.kind = 'name'
                        AND name_fact.key IS NULL
                        AND name_fact.seq = (
                            SELECT MAX(f.seq) FROM facts AS f
                            WHERE f.session_id = s.id AND f.kind = 'name' AND f.key IS NULL
                        )
                    WHERE s.cwd = $1 ORDER BY s.created_at DESC
                    """,
                    options["cwd"],
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT s.id, s.created_at, s.cwd, s.parent_session_id, s.metadata,
                           name_fact.value IS NOT NULL AS has_session_name
                    FROM sessions AS s
                    LEFT JOIN facts AS name_fact
                        ON name_fact.session_id = s.id AND name_fact.kind = 'name'
                        AND name_fact.key IS NULL
                        AND name_fact.seq = (
                            SELECT MAX(f.seq) FROM facts AS f
                            WHERE f.session_id = s.id AND f.kind = 'name' AND f.key IS NULL
                        )
                    ORDER BY s.created_at DESC
                    """
                )
            return [_decode_session_row(row, self._path(row["id"])) for row in rows]
        finally:
            await self.pool.release(conn)

    async def delete(self, metadata: SessionMetadata) -> None:
        session_id = metadata["id"]
        async with self._ops_lock:
            storage = self._active_storages.get(session_id)
            if storage is not None:
                await storage.release()
                self._active_storages.pop(session_id, None)
            conn = await self._acquire()
            try:
                async with conn.transaction():
                    for table in (
                        "branch_entries",
                        "session_entries",
                        "entry_materialized",
                        "session_materialized",
                        "records",
                        "lane_moves",
                        "facts",
                        "branch_tips",
                        "lanes",
                    ):
                        await conn.execute(f"DELETE FROM {table} WHERE session_id = $1", session_id)
                    await conn.execute(
                        "DELETE FROM writer_leases WHERE session_id = $1", session_id
                    )
                    await conn.execute(
                        "DELETE FROM session_stats WHERE session_id = $1", session_id
                    )
                    await conn.execute(
                        "DELETE FROM session_sequences WHERE session_id = $1", session_id
                    )
                    await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
            finally:
                await self.pool.release(conn)

    async def fork(
        self,
        source: SessionMetadata,
        options: ForkOptions | None = None,
    ) -> Session:
        options = options or {}
        fork_options = cast(dict[str, Any], options)
        session_id = options.get("id") or uuidv7()
        async with self._ops_lock:
            conn = await self._acquire()
            try:
                async with conn.transaction():
                    source_row = await self._require_session(conn, source["id"])
                    existing = await conn.fetchval(
                        "SELECT 1 FROM sessions WHERE id = $1", session_id
                    )
                    if existing is not None:
                        raise SessionError(
                            "already_exists", f"Session already exists: {session_id}"
                        )
                    entries: list[asyncpg.Record] = []
                    lanes: list[LanePointer] = []
                    branch_tips: list[str] = []
                    branch_fork_target: str | None = None
                    if options.get("scope") == "tree":
                        entries = await conn.fetch(
                            """
                            SELECT id, entry_seq, parent_id, type, timestamp, payload
                            FROM session_entries WHERE session_id = $1 ORDER BY entry_seq
                            """,
                            source["id"],
                        )
                        lanes = await _read_lanes(conn, source["id"])
                        branch_tips = [
                            row["tip_id"]
                            for row in await conn.fetch(
                                "SELECT tip_id FROM branch_tips WHERE session_id = $1 ORDER BY tip_id",
                                source["id"],
                            )
                        ]
                    else:
                        lane_row = await conn.fetchrow(
                            "SELECT leaf_id FROM lanes WHERE session_id = $1 AND lane = 'main'",
                            source["id"],
                        )
                        selected = options.get("entryId")
                        if selected is None:
                            selected = lane_row["leaf_id"] if lane_row is not None else None
                        if selected is not None:
                            target = await conn.fetchrow(
                                """
                                SELECT id, parent_id, type FROM session_entries
                                WHERE session_id = $1 AND id = $2
                                """,
                                source["id"],
                                selected,
                            )
                            if target is None or target["type"] != "message":
                                raise SessionError(
                                    "invalid_fork_target",
                                    f"Fork target is not a message entry: {selected}",
                                )
                            position = options.get("position") or (
                                "at" if options.get("entryId") is None else "before"
                            )
                            branch_fork_target = (
                                target["id"] if position == "at" else target["parent_id"]
                            )
                        lanes = [{"lane": "main", "leafId": branch_fork_target}]
                        if branch_fork_target is not None:
                            branch = await _read_cached_branch(
                                conn, source["id"], branch_fork_target
                            )
                            if branch is None:
                                raise SessionError(
                                    "invalid_fork_target",
                                    f"Fork target is not on a cached branch: {branch_fork_target}",
                                )
                            rows = await _query_cached_branch_rows(
                                conn, source["id"], branch, {"order": "oldestFirst"}
                            )
                            entries = rows
                            branch_tips = [branch_fork_target]
                    copied_ids = {row["id"] for row in entries}
                    latest_name = await _read_latest_fact(conn, source["id"], "name", None)
                    label_rows = await _read_latest_label_facts(conn, source["id"])
                    labels_to_copy = [
                        row
                        for row in label_rows
                        if options.get("scope") == "tree"
                        or (row["key"] is not None and row["key"] in copied_ids)
                    ]
                    created_at = _now_ms()
                    await conn.execute(
                        """
                        INSERT INTO sessions (id, created_at, cwd, parent_session_id, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        session_id,
                        _iso_from_ms(created_at),
                        fork_options.get("cwd") or source_row["cwd"],
                        options.get("parentSessionId") or source["id"],
                        _json_dumps(fork_options["metadata"])
                        if fork_options.get("metadata") is not None
                        else None,
                    )
                    message_count = sum(1 for row in entries if row["type"] == "message")
                    await conn.execute(
                        """
                        INSERT INTO session_stats (
                            session_id, message_count, cached_tokens, uncached_tokens,
                            total_tokens, cost_total
                        ) VALUES ($1, $2, 0, 0, 0, 0)
                        """,
                        session_id,
                        message_count,
                    )
                    next_seq = 1
                    for row in entries:
                        entry = _decode_entry(row)
                        entry["seq"] = next_seq
                        await _insert_entry_row(conn, session_id, seq=next_seq, entry=entry)
                        next_seq += 1
                    if options.get("scope") == "tree":
                        for lane in lanes:
                            await _insert_lane(
                                conn, session_id, next_seq, lane["lane"], lane["leafId"]
                            )
                            next_seq += 1
                    else:
                        await conn.execute(
                            """
                            INSERT INTO lanes (session_id, lane, leaf_id, open_operation_id)
                            VALUES ($1, 'main', $2, NULL)
                            """,
                            session_id,
                            branch_fork_target,
                        )
                    if latest_name is not None and latest_name["value"] is not None:
                        await _append_fact(
                            conn,
                            session_id,
                            next_seq,
                            "name",
                            None,
                            latest_name["value"],
                        )
                        next_seq += 1
                    for label in labels_to_copy:
                        await _append_fact(
                            conn, session_id, next_seq, "label", label["key"], label["value"]
                        )
                        next_seq += 1
                    await _set_seq(conn, session_id, next_seq)
                    for tip in branch_tips:
                        await _build_cached_branch(conn, session_id, tip)
                    metadata: PgSessionMetadata = {
                        "id": session_id,
                        "createdAt": created_at,
                        "cwd": fork_options.get("cwd") or source_row["cwd"],
                        "path": self._path(session_id),
                    }
                    if options.get("parentSessionId") is not None:
                        metadata["parentSessionId"] = options["parentSessionId"]
                    if fork_options.get("metadata") is not None:
                        metadata["metadata"] = fork_options["metadata"]
                    return await self._claim_with_conn(conn, metadata)
            finally:
                await self.pool.release(conn)

    async def repair_branch_cache(self, metadata: SessionMetadata) -> None:
        async with self._ops_lock:
            storage = self._active_storages.get(metadata["id"])
            if storage is not None:
                await storage.release()
                self._active_storages.pop(metadata["id"], None)
            conn = await self._acquire()
            try:
                async with conn.transaction():
                    await self._require_session(conn, metadata["id"])
                    lease = await _acquire_lease(conn, metadata["id"], uuidv7())
                    if lease is None:
                        raise SessionError(
                            "storage",
                            f"PG session {metadata['id']} already has an active writer",
                        )
                    try:
                        await _rebuild_branch_cache(conn, metadata["id"])
                    finally:
                        await _release_lease(conn, metadata["id"], lease)
            finally:
                await self.pool.release(conn)


class PgSessionSearch:
    """PostgreSQL v4 会话搜索（entries + records + facts 的 payload）。"""

    def __init__(
        self,
        repo: PostgresV4SessionRepo,
    ) -> None:
        self._repo = repo

    async def search(self, options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        options = options or {}
        text = (options.get("text") or "").strip()
        if not text:
            return []
        pattern = f"%{text}%"
        conn = await self._repo._acquire()
        try:
            rows = await conn.fetch(
                """
                SELECT kind, session_id, entry_id, timestamp, snippet, cwd, created_at
                FROM (
                    SELECT 'entry' AS kind, e.session_id, e.id AS entry_id,
                           e.timestamp AS timestamp, e.payload AS snippet
                    FROM session_entries AS e WHERE e.payload ILIKE $1
                    UNION ALL
                    SELECT 'record' AS kind, r.session_id, r.id AS entry_id,
                           r.timestamp AS timestamp, r.payload AS snippet
                    FROM records AS r WHERE r.payload ILIKE $1
                    UNION ALL
                    SELECT 'fact' AS kind, f.session_id,
                           ('fact:' || f.kind || ':' || COALESCE(f.key, '') || ':' || f.seq)
                               AS entry_id,
                           f.seq::text AS timestamp, COALESCE(f.value, '') AS snippet
                    FROM facts AS f WHERE COALESCE(f.value, '') ILIKE $1
                ) AS hits
                JOIN sessions AS s ON s.id = hits.session_id
                WHERE $2::text IS NULL OR s.cwd = $2
                ORDER BY s.created_at DESC
                """,
                pattern,
                options.get("cwd"),
            )
            result: list[dict[str, Any]] = []
            for row in rows:
                metadata = _decode_session_row(
                    {
                        "id": row["session_id"],
                        "created_at": row["created_at"],
                        "cwd": row["cwd"],
                        "parent_session_id": None,
                        "metadata": None,
                        "has_session_name": False,
                    },
                    self._repo._path(row["session_id"]),
                )
                result.append(
                    {
                        "metadata": metadata,
                        "entryId": row["entry_id"],
                        "timestamp": str(row["timestamp"]),
                        "snippet": row["snippet"],
                    }
                )
            return result
        finally:
            await self._repo.pool.release(conn)


def create_postgres_v4_repo(
    dsn: str | None = None,
    *,
    schema: str | None = None,
    min_size: int = 1,
    max_size: int = 10,
) -> PostgresV4SessionRepo:
    """创建 Postgres v4 会话仓库（需先调用 connect()）。"""
    return PostgresV4SessionRepo(dsn, schema=schema, min_size=min_size, max_size=max_size)


__all__ = [
    "create_postgres_v4_repo",
    "PgSessionCreateOptions",
    "PgSessionMetadata",
    "PgSessionSearch",
    "PostgresV4SessionRepo",
    "PostgresV4SessionStorage",
]
