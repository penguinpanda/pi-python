"""v4 Session 后端一致性用例（对齐 TS `harness/session/testing/conformance.ts`）。

提供与后端无关的 conformance 工厂：任意实现 `SessionRepo` 的存储
（InMemory / JSONL / PostgreSQL）都可复用同一组行为校验。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from ..types import SessionError


def _zero_usage() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total": 0,
        },
    }


def create_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def create_assistant_message(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": _zero_usage(),
        "stopReason": "stop",
        "timestamp": 1,
    }


def operation_started(record_id: str, lane: str, kind: str) -> dict[str, Any]:
    intent: dict[str, Any]
    if kind == "run":
        intent = {"kind": "run", "originalPrompt": [], "initialMessages": []}
    elif kind == "compaction":
        intent = {"kind": "compaction", "resultEntryId": f"{record_id}-result"}
    else:
        intent = {"kind": "navigation", "targetId": None, "summarize": False}
    return {
        "type": "operation_started",
        "id": record_id,
        "lane": lane,
        "sourceLeafId": None,
        "intent": intent,
    }


def entry_ids(entries: list[dict]) -> list[str]:
    return [entry["id"] for entry in entries]


async def rejects_with_code(operation, code: str) -> None:
    try:
        await operation
    except SessionError as error:
        if error.code == code:
            return
        raise AssertionError(
            f"Expected SessionError code {code}, got {error.code}: {error}"
        ) from error
    raise AssertionError(f"Expected SessionError code {code}, but operation succeeded")


async def _case_entries_and_lanes_assigns_parents_and_one_sequence(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    root = await session.append_entry(
        {"type": "message", "id": "root", "message": create_user_message("root")},
        "main",
    )
    await session.create_lane("thread", root["id"])
    child = await session.append_entry(
        {
            "type": "custom",
            "id": "child",
            "customType": "note",
            "data": {"value": 1},
        },
        "thread",
    )
    record = await session.append_record(operation_started("run", "thread", "run"))
    await session.set_name("Example")
    await session.set_label(root["id"], "checkpoint")
    await session.move_lane("main", child["id"])
    assert (root["parentId"], root["seq"]) == (None, 1)
    assert (child["parentId"], child["seq"]) == ("root", 3)
    assert record["seq"] == 4
    for timestamp in (root["timestamp"], child["timestamp"], record["timestamp"]):
        assert isinstance(timestamp, int) and timestamp >= 0
    assert [(item["kind"], item["seq"]) for item in await session.get_log()] == [
        ("entry", 1),
        ("lane", 2),
        ("entry", 3),
        ("record", 4),
        ("fact", 5),
        ("fact", 6),
        ("lane", 7),
    ]
    assert await session.get_lanes() == [
        {"lane": "main", "leafId": "child"},
        {"lane": "thread", "leafId": "child"},
    ]


async def _case_entries_and_lanes_rejects_duplicate_ids_without_changing_state(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    await session.append_entry(
        {"type": "message", "id": "shared", "message": create_user_message("root")},
        "main",
    )
    await rejects_with_code(
        session.append_record(operation_started("shared", "main", "run")),
        "already_exists",
    )
    await session.append_record(operation_started("run", "main", "run"))
    await rejects_with_code(
        session.append_entry({"type": "custom", "id": "run", "customType": "note"}, "main"),
        "already_exists",
    )
    assert [item["seq"] for item in await session.get_log()] == [1, 2]


async def _case_entries_and_lanes_isolates_lanes_while_sharing_the_tree(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    root = await session.append_entry(
        {"type": "message", "id": "root", "message": create_user_message("root")},
        "main",
    )
    await session.create_lane("thread", root["id"])
    await session.append_entry(
        {"type": "message", "id": "main-child", "message": create_user_message("main")},
        "main",
    )
    await session.append_entry(
        {
            "type": "message",
            "id": "thread-child",
            "message": create_user_message("thread"),
        },
        "thread",
    )
    assert await session.get_lanes() == [
        {"lane": "main", "leafId": "main-child"},
        {"lane": "thread", "leafId": "thread-child"},
    ]
    assert entry_ids(
        await session.find_entries_on_branch({"start": "main-child", "order": "oldestFirst"})
    ) == ["root", "main-child"]
    assert entry_ids(
        await session.find_entries_on_branch({"start": "thread-child", "order": "oldestFirst"})
    ) == ["root", "thread-child"]


async def _case_entries_and_lanes_validates_lane_lifecycle_and_targets(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    await rejects_with_code(session.create_lane("main", None), "already_exists")
    await rejects_with_code(session.create_lane("thread", "missing"), "not_found")
    await rejects_with_code(session.move_lane("missing", None), "invalid_lane")


async def _case_entries_and_lanes_binds_lane_views_without_caching_leaves(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    root = await session.append_message(create_user_message("root"))
    await session.create_lane("thread", root)
    thread = session.view("thread")
    main_child, thread_child = await asyncio.gather(
        session.append_message(create_user_message("main")),
        thread.append_message(create_user_message("thread")),
    )
    assert await session.get_leaf_id() == main_child
    assert await thread.get_leaf_id() == thread_child
    assert entry_ids(await session.find_entries_on_branch({"order": "oldestFirst"})) == [
        root,
        main_child,
    ]
    assert entry_ids(await thread.find_entries_on_branch({"order": "oldestFirst"})) == [
        root,
        thread_child,
    ]
    empty = await repo.create({"id": "empty"})
    assert await empty.find_entries_on_branch() == []


async def _case_entries_and_lanes_appends_provisioned_entries_with_existing_ids(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    entry = await session.append_entry(
        {
            "type": "custom",
            "id": "provisioned",
            "customType": "note",
            "data": {"value": 1},
        },
        "main",
    )
    assert entry["customType"] == "note"
    assert (entry["id"], entry["parentId"], entry["seq"]) == (
        "provisioned",
        None,
        1,
    )
    assert await session.get_leaf_id() == "provisioned"


async def _case_entries_and_lanes_persists_tool_result_termination_decisions(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    entry = await session.append_entry(
        {
            "type": "message",
            "id": "tool-result",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "toolName": "example",
                "content": [{"type": "text", "text": "done"}],
                "isError": False,
                "timestamp": 1,
            },
            "terminate": True,
        },
        "main",
    )
    assert entry["terminate"] is True
    stored = await session.get_entry(entry["id"])
    assert stored is not None and stored["type"] == "message"
    assert stored["terminate"] is True
    assert await session.find_entries() == [entry]
    assert await session.get_log() == [{"kind": "entry", "seq": entry["seq"], "entry": entry}]


async def _case_entries_and_lanes_linearizes_concurrent_writes_across_two_lanes(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    root = await session.append_entry(
        {"type": "message", "id": "root", "message": create_user_message("root")},
        "main",
    )
    await session.create_lane("thread", root["id"])
    completion_order: list[str] = []

    async def write(entry_id: str, lane: str) -> dict[str, Any]:
        entry = await session.append_entry(
            {"type": "custom", "id": entry_id, "customType": "note"}, lane
        )
        completion_order.append(entry["id"])
        return entry

    entries = await asyncio.gather(
        write("main-1", "main"),
        write("thread-1", "thread"),
        write("main-2", "main"),
        write("thread-2", "thread"),
    )
    commit_order = [entry["id"] for entry in sorted(entries, key=lambda e: e["seq"])]
    assert len({entry["seq"] for entry in entries}) == len(entries)
    assert completion_order == commit_order
    concurrent_ids = {entry["id"] for entry in entries}
    assert [
        item["entry"]["id"]
        for item in await session.get_log()
        if item["kind"] == "entry" and item["entry"]["id"] in concurrent_ids
    ] == commit_order
    sequences = [item["seq"] for item in await session.get_log()]
    assert sequences == sorted(sequences)


async def _case_records_and_log_commits_records_and_lane_moves_as_separate_mutations(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    root = await session.append_entry(
        {"type": "message", "id": "root", "message": create_user_message("root")},
        "main",
    )
    finished = await session.append_record(
        {
            "type": "operation_finished",
            "id": "finish",
            "lane": "main",
            "runId": "run",
            "outcome": "completed",
        }
    )
    assert finished["seq"] == 2
    assert await session.get_lanes() == [{"lane": "main", "leafId": "root"}]
    await session.move_lane("main", None)
    assert await session.get_lanes() == [{"lane": "main", "leafId": None}]
    assert await session.get_log() == [
        {"kind": "entry", "seq": 1, "entry": root},
        {"kind": "record", "seq": 2, "record": finished},
        {"kind": "lane", "seq": 3, "lane": "main", "leafId": None},
    ]
    await rejects_with_code(session.move_lane("main", "missing"), "not_found")
    assert len(await session.find_records()) == 1
    assert [item["seq"] for item in await session.get_log()] == [1, 2, 3]


async def _case_records_and_log_keeps_lane_names_permanent_with_their_recovery_records(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    await session.create_lane("thread", None)
    await session.append_record(operation_started("old-run", "thread", "run"))
    await session.append_record(
        {
            "type": "queue_enqueued",
            "id": "old-next-run",
            "lane": "thread",
            "queue": "nextRun",
            "target": {
                "type": "message",
                "id": "queued-message",
                "message": create_user_message("queued"),
            },
        }
    )
    assert [r["id"] for r in await session.find_records({"lane": "thread"})] == [
        "old-next-run",
        "old-run",
    ]
    assert [
        item["record"]["id"] for item in await session.get_log() if item["kind"] == "record"
    ] == ["old-run", "old-next-run"]
    await rejects_with_code(session.create_lane("thread", None), "already_exists")


async def _case_records_and_log_persists_queue_cancellation_without_consuming_target(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    enqueued = await session.append_record(
        {
            "type": "queue_enqueued",
            "id": "enqueue",
            "lane": "main",
            "queue": "nextRun",
            "target": {
                "type": "message",
                "id": "queued-message",
                "message": create_user_message("queued"),
            },
        }
    )
    cancelled = await session.append_record(
        {
            "type": "queue_cancelled",
            "id": "cancel",
            "lane": "main",
            "entryId": "queued-message",
        }
    )
    assert (cancelled["seq"], cancelled["entryId"]) == (2, "queued-message")
    assert "runId" not in cancelled
    assert await session.get_entry("queued-message") is None
    cancellations = await session.find_records({"type": "queue_cancelled"})
    assert cancellations[0]["entryId"] == "queued-message"
    assert cancellations == [cancelled]
    assert await session.get_log() == [
        {"kind": "record", "seq": enqueued["seq"], "record": enqueued},
        {"kind": "record", "seq": cancelled["seq"], "record": cancelled},
    ]


async def _case_records_and_log_filters_records_by_lane_type_run_sequence_and_order(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    await session.append_record(operation_started("run-1", "main", "run"))
    await session.append_record(
        {
            "type": "step_attempt",
            "id": "attempt-1",
            "lane": "main",
            "runId": "run-1",
            "step": "assistant",
            "attempt": 1,
            "resultEntryId": "assistant-1",
        }
    )
    await session.create_lane("thread", None)
    await session.append_record(operation_started("run-2", "thread", "run"))
    await session.append_record(
        {
            "type": "step_attempt",
            "id": "attempt-2",
            "lane": "thread",
            "runId": "run-2",
            "step": "assistant",
            "attempt": 1,
            "resultEntryId": "assistant-2",
        }
    )
    assert [r["id"] for r in await session.find_records({"lane": "thread"})] == [
        "attempt-2",
        "run-2",
    ]
    assert [
        r["id"]
        for r in await session.find_records({"type": "step_attempt", "order": "oldestFirst"})
    ] == ["attempt-1", "attempt-2"]
    assert [r["id"] for r in await session.find_records({"runId": "run-1", "afterSeq": 1})] == [
        "attempt-1"
    ]
    assert [r["id"] for r in await session.find_records({"limit": 1})] == ["attempt-2"]


async def _case_records_and_log_filters_operation_starts_by_operation_kind(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    for record_id, kind in [
        ("run-old", "run"),
        ("compaction", "compaction"),
        ("navigation", "navigation"),
        ("run-new", "run"),
    ]:
        await session.append_record(operation_started(record_id, "main", kind))
        await session.append_record(
            {
                "type": "operation_finished",
                "id": f"{record_id}-finished",
                "lane": "main",
                "runId": record_id,
                "outcome": "completed",
            }
        )
    assert [
        r["id"]
        for r in await session.find_records(
            {"type": "operation_started", "operationKind": "run", "order": "oldestFirst"}
        )
    ] == ["run-old", "run-new"]
    assert [
        r["id"]
        for r in await session.find_records(
            {"type": "operation_started", "operationKind": "compaction"}
        )
    ] == ["compaction"]
    assert [
        r["id"]
        for r in await session.find_records(
            {"type": "operation_started", "operationKind": "navigation"}
        )
    ] == ["navigation"]
    assert [
        r["id"]
        for r in await session.find_records(
            {"type": "operation_started", "operationKind": "run", "limit": 1}
        )
    ] == ["run-new"]


async def _case_records_and_log_tracks_and_enforces_one_open_operation_per_lane(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    assert await session.find_open_operations("main", {"limit": 2}) == []
    first = await session.append_record(operation_started("first", "main", "run"))
    assert await session.find_open_operations("main", {"limit": 2}) == [first]
    await rejects_with_code(
        session.append_record(operation_started("second", "main", "run")),
        "storage",
    )
    assert await session.find_open_operations("main", {"limit": 2}) == [first]
    await session.append_record(
        {
            "type": "operation_finished",
            "id": "finish-first",
            "lane": "main",
            "runId": first["id"],
            "outcome": "completed",
        }
    )
    assert await session.find_open_operations("main", {"limit": 2}) == []


async def _case_records_and_log_earlier_finish_does_not_close_later_start(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    await session.append_record(
        {
            "type": "operation_finished",
            "id": "finish-before-start",
            "lane": "main",
            "runId": "run",
            "outcome": "completed",
        }
    )
    started = await session.append_record(operation_started("run", "main", "run"))
    assert await session.find_open_operations("main", {"limit": 2}) == [started]


async def _case_records_and_log_scopes_open_operations_by_lane_and_limit(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    await session.create_lane("thread", None)
    main_run = await session.append_record(operation_started("main-run", "main", "run"))
    thread_navigation = await session.append_record(
        operation_started("thread-navigation", "thread", "navigation")
    )
    assert await session.find_open_operations("main") == [main_run]
    assert await session.find_open_operations("main", {"limit": 1}) == [main_run]
    assert await session.find_open_operations("thread", {"limit": 2}) == [thread_navigation]


async def _case_records_and_log_returns_immutable_open_operation_records(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    committed = await session.append_record(operation_started("run", "main", "run"))
    (read,) = await session.find_open_operations("main")
    assert read["intent"]["kind"] == "run"
    read["intent"]["originalPrompt"].append(create_user_message("mutated"))
    assert await session.find_open_operations("main") == [committed]


async def _case_queries_and_facts_rejects_invalid_queries_before_empty_reads(repo: Any) -> None:
    session = await repo.create({"id": "invalid-queries"})
    await session.create_lane("thread", None)
    thread = session.view("thread")
    await rejects_with_code(session.find_entries({"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_entry({"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_entries_on_branch({"limit": 0}), "invalid_query")
    await rejects_with_code(
        thread.find_entries_on_branch({"cursor": {"afterSeq": -1}}),
        "invalid_query",
    )
    await rejects_with_code(thread.find_entry_on_branch({"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_records({"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_records({"operationKind": "run"}), "invalid_query")
    await rejects_with_code(
        session.find_records({"type": "step_attempt", "operationKind": "run"}),
        "invalid_query",
    )
    await rejects_with_code(session.find_open_operations("main", {"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_open_operations("main", {"limit": -1}), "invalid_query")
    await rejects_with_code(session.get_log({"afterSeq": -1}), "invalid_query")


async def _case_queries_and_facts_supports_bounded_filtered_and_cursor_queries(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    await session.append_entry(
        {"type": "message", "id": "root", "message": create_user_message("root")},
        "main",
    )
    await session.append_entry(
        {"type": "custom", "id": "old-note", "customType": "note", "data": 1},
        "main",
    )
    await session.append_entry(
        {
            "type": "compaction",
            "id": "compact",
            "summary": "summary",
            "retainedTail": [],
            "tokensBefore": 10,
        },
        "main",
    )
    await session.append_entry(
        {"type": "custom", "id": "new-note", "customType": "note", "data": 2},
        "main",
    )
    tail = await session.append_entry(
        {
            "type": "message",
            "id": "tail",
            "message": create_assistant_message("tail"),
        },
        "main",
    )
    assert entry_ids(await session.find_entries()) == [
        "tail",
        "new-note",
        "compact",
        "old-note",
        "root",
    ]
    assert entry_ids(
        await session.find_entries({"order": "oldestFirst", "cursor": {"afterSeq": 2}, "limit": 2})
    ) == ["compact", "new-note"]
    assert entry_ids(await session.find_entries({"customType": "note"})) == [
        "new-note",
        "old-note",
    ]
    assert entry_ids(
        await session.find_entries_on_branch({"start": "tail", "customType": "note", "limit": 1})
    ) == ["new-note"]
    assert entry_ids(
        await session.find_entries_on_branch(
            {"start": "tail", "stopAtType": "compaction", "type": "message"}
        )
    ) == ["tail"]
    assert (
        entry_ids(
            await session.find_entries_on_branch(
                {"start": "tail", "stopAtId": "tail", "type": "custom"}
            )
        )
        == []
    )
    assert entry_ids(
        await session.find_entries_on_branch(
            {"start": "tail", "stopAtType": "custom", "order": "oldestFirst"}
        )
    ) == ["root", "old-note"]
    await rejects_with_code(session.find_entries({"limit": 0}), "invalid_query")
    await rejects_with_code(session.find_entries_on_branch({"start": "missing"}), "not_found")
    assert tail["id"] == "tail"


async def _case_queries_and_facts_keeps_latest_value_facts_and_computes_stats(repo: Any) -> None:
    session = await repo.create({"id": "session"})
    assistant = create_assistant_message("answer")
    assistant["usage"] = {
        "input": 10,
        "output": 5,
        "cache_read": 3,
        "cache_write": 2,
        "total_tokens": 20,
        "cost": {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4, "total": 10},
    }
    await session.append_entry(
        {"type": "message", "id": "user", "message": create_user_message("question")},
        "main",
    )
    assistant_entry = await session.append_entry(
        {"type": "message", "id": "assistant", "message": assistant}, "main"
    )
    await session.append_record(
        {
            "type": "usage",
            "id": "assistant-usage",
            "lane": "main",
            "cause": "assistant",
            "runId": "run",
            "entryId": "assistant",
            "attempt": 1,
            "stopReason": "stop",
            "usage": assistant["usage"],
        }
    )
    await session.append_record(
        {
            "type": "usage",
            "id": "deferred-usage",
            "lane": "main",
            "cause": "deferred_fetch",
            "runId": "run",
            "entryId": "deferred-result",
            "attempt": 1,
            "stopReason": "deferred",
            "usage": _zero_usage(),
        }
    )
    await session.create_lane("thread", assistant_entry["id"])
    await session.append_record(
        {
            "type": "usage",
            "id": "correction",
            "lane": "thread",
            "cause": "adjustment",
            "details": {"reason": "provider correction"},
            "usage": {
                "input": -2,
                "output": 0,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": -2,
                "cost": {
                    "input": -0.5,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "total": -0.5,
                },
            },
        }
    )
    await session.set_name("First")
    await session.set_name("Second")
    await session.set_label("user", "keep")
    await session.set_label("user", None)
    await rejects_with_code(session.set_label("missing", "checkpoint"), "not_found")
    assert await session.get_name() == "Second"
    assert await session.get_label("user") is None
    usage_records = await session.find_records({"type": "usage", "order": "oldestFirst"})
    assert [r["cause"] for r in usage_records] == [
        "assistant",
        "deferred_fetch",
        "adjustment",
    ]
    deferred_usage = next(r for r in usage_records if r["cause"] == "deferred_fetch")
    assert deferred_usage["stopReason"] == "deferred"
    assert await session.get_stats() == {
        "messageCount": 2,
        "cachedTokens": 3,
        "uncachedTokens": 10,
        "totalTokens": 18,
        "costTotal": 9.5,
    }


async def _case_validation_and_immutability_returns_immutable_copies_from_reads(repo: Any) -> None:
    session = await repo.create({"id": "immutable"})
    metadata = await session.get_metadata()
    data = {"nested": {"value": 1}}
    await session.append_entry(
        {"type": "custom", "id": "custom", "customType": "note", "data": data},
        "main",
    )
    data["nested"]["value"] = 50
    read = await session.get_entry("custom")
    assert read is not None
    read["data"]["nested"]["value"] = 99
    read_metadata = await session.get_metadata()
    read_metadata["id"] = "changed"
    log = await session.get_log()
    assert log[0]["kind"] == "entry" and log[0]["entry"]["type"] == "custom"
    log[0]["entry"]["data"]["nested"]["value"] = 100
    assert await session.get_metadata() == metadata
    assert await session.get_entry("custom") == {
        "type": "custom",
        "id": "custom",
        "customType": "note",
        "data": {"nested": {"value": 1}},
        "parentId": None,
        "seq": 1,
        "timestamp": read["timestamp"],
    }


async def _case_validation_and_immutability_rejects_non_json_entries_before_storage_mutation(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    cyclic: dict = {}
    cyclic["self"] = cyclic
    data: Any
    for data in (
        {"value": float("nan")},
        [float("nan")],
        {"value": 1 + 2j},
        {"value": set()},
        cyclic,
    ):
        await rejects_with_code(session.append_custom_entry("invalid", data), "invalid_payload")
    assert await session.get_leaf_id() is None
    assert await session.find_entries() == []
    assert await session.get_log() == []
    valid_id = await session.append_custom_entry("valid", {"value": 1})
    assert (await session.get_entry(valid_id))["seq"] == 1


async def _case_validation_and_immutability_rejects_non_json_records_before_storage_mutation(
    repo: Any,
) -> None:
    session = await repo.create({"id": "session"})
    for record_id, value in (
        ("nan-record", float("nan")),
        ("complex-record", 1 + 2j),
        ("set-record", set()),
    ):
        await rejects_with_code(
            session.append_record(
                {
                    "type": "tool_started",
                    "id": record_id,
                    "lane": "main",
                    "runId": "run",
                    "assistantEntryId": "assistant",
                    "toolIndex": 0,
                    "toolCallId": "call",
                    "toolName": "example",
                    "effectiveArgs": {"value": value},
                    "resultEntryId": "result",
                    "replay": "never",
                }
            ),
            "invalid_payload",
        )
    assert await session.find_records() == []
    assert await session.get_log() == []
    assert (await session.append_record(operation_started("valid-record", "main", "run")))[
        "seq"
    ] == 1


async def _case_repository_and_forks_creates_lists_and_opens_sessions(repo: Any) -> None:
    session = await repo.create({"id": "one"})
    entry_id = await session.append_message(create_user_message("persisted"))
    metadata = await session.get_metadata()
    listed = await repo.list()
    assert len(listed) == 1
    assert listed[0]["id"] == metadata["id"]
    assert listed[0]["createdAt"] == metadata["createdAt"]
    assert listed[0].get("parentSessionId") == metadata.get("parentSessionId")
    assert entry_ids(await (await repo.open(metadata)).find_entries()) == [entry_id]
    await rejects_with_code(repo.create({"id": "one"}), "already_exists")


async def _case_repository_and_forks_deletes_sessions_idempotently(repo: Any) -> None:
    session = await repo.create({"id": "one"})
    metadata = await session.get_metadata()
    await repo.delete(metadata)
    await rejects_with_code(repo.open(metadata), "not_found")
    await repo.delete(metadata)


async def _case_repository_and_forks_forks_one_branch_with_selected_facts_and_no_records(
    repo: Any,
) -> None:
    source = await repo.create({"id": "source"})
    root = await source.append_message(create_user_message("root"))
    shared = await source.append_message(create_assistant_message("shared"))
    await source.create_lane("thread", shared)
    thread_child = await source.view("thread").append_message(create_user_message("thread"))
    main_child = await source.append_message(create_user_message("main"))
    await source.set_name("Source")
    await source.set_label(shared, "copied")
    await source.set_label(thread_child, "excluded")
    await source.append_record(operation_started("run", "main", "run"))
    await source.append_record(
        {
            "type": "usage",
            "id": "source-usage",
            "lane": "main",
            "cause": "adjustment",
            "usage": {
                "input": 10,
                "output": 5,
                "cache_read": 3,
                "cache_write": 2,
                "total_tokens": 20,
                "cost": {
                    "input": 1,
                    "output": 2,
                    "cache_read": 3,
                    "cache_write": 4,
                    "total": 10,
                },
            },
        }
    )
    fork = await repo.fork(
        await source.get_metadata(),
        {"scope": "branch", "entryId": main_child, "position": "at", "id": "branch-fork"},
    )
    assert entry_ids(await fork.find_entries({"order": "oldestFirst"})) == [
        root,
        shared,
        main_child,
    ]
    assert await fork.get_lanes() == [{"lane": "main", "leafId": main_child}]
    assert await fork.get_name() == "Source"
    assert await fork.get_label(shared) == "copied"
    assert await fork.get_label(thread_child) is None
    assert await fork.find_records() == []
    assert await fork.get_stats() == {
        "messageCount": 3,
        "cachedTokens": 0,
        "uncachedTokens": 0,
        "totalTokens": 0,
        "costTotal": 0,
    }
    await fork.append_message(create_user_message("after fork"))
    assert (await fork.get_stats())["messageCount"] == 4
    metadata = await fork.get_metadata()
    assert (metadata["id"], metadata["parentSessionId"]) == (
        "branch-fork",
        "source",
    )


async def _case_repository_and_forks_forks_a_complete_tree_with_lanes_and_facts(repo: Any) -> None:
    source = await repo.create({"id": "source"})
    root = await source.append_message(create_user_message("root"))
    await source.create_lane("thread", root)
    main_child = await source.append_message(create_user_message("main"))
    thread_child = await source.view("thread").append_message(create_user_message("thread"))
    await source.set_label(thread_child, "thread-tip")
    fork = await repo.fork(await source.get_metadata(), {"scope": "tree", "id": "tree-fork"})
    assert entry_ids(await fork.find_entries({"order": "oldestFirst"})) == [
        root,
        main_child,
        thread_child,
    ]
    assert await fork.get_lanes() == [
        {"lane": "main", "leafId": main_child},
        {"lane": "thread", "leafId": thread_child},
    ]
    assert await fork.get_label(thread_child) == "thread-tip"
    assert (await fork.get_stats())["messageCount"] == 3
    assert [item for item in await fork.get_log() if item["kind"] == "lane"] == [
        {"kind": "lane", "seq": 4, "lane": "main", "leafId": main_child},
        {"kind": "lane", "seq": 5, "lane": "thread", "leafId": thread_child},
    ]


async def _case_repository_and_forks_forks_before_an_entry_without_modifying_source(
    repo: Any,
) -> None:
    source = await repo.create({"id": "source"})
    root = await source.append_message(create_user_message("root"))
    tail = await source.append_message(create_user_message("tail"))
    fork = await repo.fork(await source.get_metadata(), {"entryId": tail, "id": "fork"})
    assert entry_ids(await fork.find_entries({"order": "oldestFirst"})) == [root]
    assert await fork.get_leaf_id() == root
    assert await source.get_leaf_id() == tail
    before_default_target = await repo.fork(
        await source.get_metadata(), {"position": "before", "id": "before-default-target"}
    )
    assert entry_ids(await before_default_target.find_entries({"order": "oldestFirst"})) == [root]
    assert await before_default_target.get_leaf_id() == root
    at_default_target = await repo.fork(
        await source.get_metadata(), {"position": "at", "id": "at-default-target"}
    )
    assert entry_ids(await at_default_target.find_entries({"order": "oldestFirst"})) == [
        root,
        tail,
    ]
    assert await at_default_target.get_leaf_id() == tail
    await rejects_with_code(
        repo.fork(await source.get_metadata(), {"entryId": "missing"}),
        "invalid_fork_target",
    )


async def _case_repository_and_forks_validates_the_default_fork_target(repo: Any) -> None:
    source = await repo.create({"id": "source-with-custom-leaf"})
    await source.append_custom_entry("not-a-message")
    await rejects_with_code(
        repo.fork(await source.get_metadata(), {"id": "fork"}),
        "invalid_fork_target",
    )


class SessionBackendFixture(Protocol):
    """一个 conformance case 独占的后端实例。"""

    @property
    def repository(self) -> Any: ...

    async def dispose(self) -> None: ...


SessionBackendFixtureFactory = Callable[[], Awaitable[SessionBackendFixture]]


@dataclass(frozen=True, slots=True)
class SessionBackendConformanceCase:
    group: str
    name: str
    run: Callable[[], Awaitable[None]]


def _make_case(
    factory: SessionBackendFixtureFactory,
    group: str,
    name: str,
    test: Callable[[Any], Awaitable[None]],
) -> SessionBackendConformanceCase:
    async def run() -> None:
        fixture = await factory()
        try:
            await test(fixture.repository)
        finally:
            await fixture.dispose()

    return SessionBackendConformanceCase(group=group, name=name, run=run)


def create_session_backend_conformance(
    factory: SessionBackendFixtureFactory,
) -> list[SessionBackendConformanceCase]:
    """返回后端一致性用例；每个 case 创建并销毁自己的 fixture。"""
    return [
        _make_case(
            factory,
            "entries and lanes",
            "assigns parents and one sequence",
            _case_entries_and_lanes_assigns_parents_and_one_sequence,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "rejects duplicate ids without changing state",
            _case_entries_and_lanes_rejects_duplicate_ids_without_changing_state,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "isolates lanes while sharing the tree",
            _case_entries_and_lanes_isolates_lanes_while_sharing_the_tree,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "validates lane lifecycle and targets",
            _case_entries_and_lanes_validates_lane_lifecycle_and_targets,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "binds lane views without caching leaves",
            _case_entries_and_lanes_binds_lane_views_without_caching_leaves,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "appends provisioned entries with existing ids",
            _case_entries_and_lanes_appends_provisioned_entries_with_existing_ids,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "persists tool result termination decisions",
            _case_entries_and_lanes_persists_tool_result_termination_decisions,
        ),
        _make_case(
            factory,
            "entries and lanes",
            "linearizes concurrent writes across two lanes",
            _case_entries_and_lanes_linearizes_concurrent_writes_across_two_lanes,
        ),
        _make_case(
            factory,
            "records and log",
            "commits records and lane moves as separate mutations",
            _case_records_and_log_commits_records_and_lane_moves_as_separate_mutations,
        ),
        _make_case(
            factory,
            "records and log",
            "keeps lane names permanent with their recovery records",
            _case_records_and_log_keeps_lane_names_permanent_with_their_recovery_records,
        ),
        _make_case(
            factory,
            "records and log",
            "persists queue cancellation without consuming target",
            _case_records_and_log_persists_queue_cancellation_without_consuming_target,
        ),
        _make_case(
            factory,
            "records and log",
            "filters records by lane type run sequence and order",
            _case_records_and_log_filters_records_by_lane_type_run_sequence_and_order,
        ),
        _make_case(
            factory,
            "records and log",
            "filters operation starts by operation kind",
            _case_records_and_log_filters_operation_starts_by_operation_kind,
        ),
        _make_case(
            factory,
            "records and log",
            "tracks and enforces one open operation per lane",
            _case_records_and_log_tracks_and_enforces_one_open_operation_per_lane,
        ),
        _make_case(
            factory,
            "records and log",
            "earlier finish does not close later start",
            _case_records_and_log_earlier_finish_does_not_close_later_start,
        ),
        _make_case(
            factory,
            "records and log",
            "scopes open operations by lane and limit",
            _case_records_and_log_scopes_open_operations_by_lane_and_limit,
        ),
        _make_case(
            factory,
            "records and log",
            "returns immutable open operation records",
            _case_records_and_log_returns_immutable_open_operation_records,
        ),
        _make_case(
            factory,
            "queries and facts",
            "rejects invalid queries before empty reads",
            _case_queries_and_facts_rejects_invalid_queries_before_empty_reads,
        ),
        _make_case(
            factory,
            "queries and facts",
            "supports bounded filtered and cursor queries",
            _case_queries_and_facts_supports_bounded_filtered_and_cursor_queries,
        ),
        _make_case(
            factory,
            "queries and facts",
            "keeps latest value facts and computes stats",
            _case_queries_and_facts_keeps_latest_value_facts_and_computes_stats,
        ),
        _make_case(
            factory,
            "validation and immutability",
            "returns immutable copies from reads",
            _case_validation_and_immutability_returns_immutable_copies_from_reads,
        ),
        _make_case(
            factory,
            "validation and immutability",
            "rejects non json entries before storage mutation",
            _case_validation_and_immutability_rejects_non_json_entries_before_storage_mutation,
        ),
        _make_case(
            factory,
            "validation and immutability",
            "rejects non json records before storage mutation",
            _case_validation_and_immutability_rejects_non_json_records_before_storage_mutation,
        ),
        _make_case(
            factory,
            "repository and forks",
            "creates lists and opens sessions",
            _case_repository_and_forks_creates_lists_and_opens_sessions,
        ),
        _make_case(
            factory,
            "repository and forks",
            "deletes sessions idempotently",
            _case_repository_and_forks_deletes_sessions_idempotently,
        ),
        _make_case(
            factory,
            "repository and forks",
            "forks one branch with selected facts and no records",
            _case_repository_and_forks_forks_one_branch_with_selected_facts_and_no_records,
        ),
        _make_case(
            factory,
            "repository and forks",
            "forks a complete tree with lanes and facts",
            _case_repository_and_forks_forks_a_complete_tree_with_lanes_and_facts,
        ),
        _make_case(
            factory,
            "repository and forks",
            "forks before an entry without modifying source",
            _case_repository_and_forks_forks_before_an_entry_without_modifying_source,
        ),
        _make_case(
            factory,
            "repository and forks",
            "validates the default fork target",
            _case_repository_and_forks_validates_the_default_fork_target,
        ),
    ]


__all__ = [
    "SessionBackendFixture",
    "SessionBackendFixtureFactory",
    "SessionBackendConformanceCase",
    "create_session_backend_conformance",
    "create_assistant_message",
    "create_user_message",
    "entry_ids",
    "operation_started",
    "rejects_with_code",
]
