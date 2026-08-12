"""record log 校验与 lane 状态归约测试（对齐 TS harness/reducer.test.ts）。"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pi_agent.session.v4.reducer import (
    EffectiveLaneConfiguration,
    LaneReductionInput,
    LaneReductionResult,
    RecordLogCorruption,
    RecordLogCorruptionReason,
    RecordLogSlice,
    reduce_lane_state,
    validate_record_log,
)
from pi_agent.session.v4.types import (
    Entry,
    LaneRecord,
    OperationFinishedRecord,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    StepAttemptRecord,
    ToolStartedRecord,
    WriteDeferredRecord,
)


def _usage() -> dict[str, Any]:
    return {
        "input": 1,
        "output": 1,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 2,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}


def assistant_message(
    content: list[dict[str, Any]],
    stop_reason: str = "stop",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "api": "openai-responses",
        "provider": "openai",
        "model": "test-model",
        "usage": _usage(),
        "stop_reason": stop_reason,
        "timestamp": 1,
    }
    if stop_reason == "deferred":
        result["deferred"] = {
            "provider": "openai",
            "model_id": "test-model",
            "api": "openai-responses",
            "id": "deferred-1",
        }
    return result


def tool_result_message(tool_call_id: str = "call-1", tool_name: str = "tool-1") -> dict[str, Any]:
    return {
        "role": "toolResult",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "content": [{"type": "text", "text": "result"}],
        "is_error": False,
        "timestamp": 1,
    }


def message_target(id: str, message: dict[str, Any]) -> dict[str, Any]:
    return {"type": "message", "id": id, "message": message}


def persisted_entry(
    target: dict[str, Any], seq: int, parent_id: str | None = None
) -> dict[str, Any]:
    return {**target, "parentId": parent_id, "seq": seq, "timestamp": seq}


def run_started(seq: int = 1, **options: Any) -> OperationStartedRecord:
    return {
        "type": "operation_started",
        "id": options.get("id", "run-1"),
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "sourceLeafId": None,
        "intent": {
            "kind": "run",
            "originalPrompt": [],
            "initialMessages": options.get("initialMessages", []),
        },
    }


def compaction_started(seq: int, result_entry_id: str = "compaction-1") -> OperationStartedRecord:
    return {
        "type": "operation_started",
        "id": "compact-1",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "sourceLeafId": "source",
        "intent": {"kind": "compaction", "resultEntryId": result_entry_id},
    }


def attempt(
    seq: int,
    run_id: str,
    step: str,
    attempt_number: int,
    result_entry_id: str,
    compaction_reason: str | None = None,
) -> StepAttemptRecord:
    base: dict[str, Any] = {
        "type": "step_attempt",
        "id": f"attempt-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "runId": run_id,
        "attempt": attempt_number,
        "resultEntryId": result_entry_id,
    }
    if step == "compaction":
        base["step"] = "compaction"
        base["compactionReason"] = compaction_reason or "manual"
    else:
        base["step"] = step
    return base  # type: ignore[return-value]


def abort_requested(seq: int, run_id: str = "run-1") -> LaneRecord:
    return {
        "type": "abort_requested",
        "id": f"abort-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "runId": run_id,
    }


def operation_finished(
    seq: int,
    run_id: str = "run-1",
    outcome: str = "completed",
) -> OperationFinishedRecord:
    return {
        "type": "operation_finished",
        "id": f"finish-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "runId": run_id,
        "outcome": outcome,
    }


def tool_started(seq: int, **overrides: Any) -> ToolStartedRecord:
    base: dict[str, Any] = {
        "type": "tool_started",
        "id": f"tool-start-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "runId": "run-1",
        "assistantEntryId": "assistant-tools",
        "toolIndex": 0,
        "toolCallId": "call-1",
        "toolName": "tool-1",
        "effectiveArgs": {},
        "resultEntryId": "tool-result-1",
        "replay": "never",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def queue_enqueued(
    seq: int,
    target: dict[str, Any] | None = None,
    queue: str = "steer",
) -> QueueEnqueuedRecord:
    base: dict[str, Any] = {
        "type": "queue_enqueued",
        "id": f"queue-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "target": target or message_target("queue-1", user_message("queued")),
    }
    if queue == "nextRun":
        base["queue"] = "nextRun"
    else:
        base["queue"] = queue
        base["runId"] = "run-1"
    return base  # type: ignore[return-value]


def queue_cancelled(
    seq: int, entry_id: str = "queue-1", run_id: str | None = "run-1"
) -> QueueCancelledRecord:
    base: dict[str, Any] = {
        "type": "queue_cancelled",
        "id": f"cancel-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "entryId": entry_id,
    }
    if run_id is not None:
        base["runId"] = run_id
    return base  # type: ignore[return-value]


def write_deferred(
    seq: int,
    target: dict[str, Any] | None = None,
) -> WriteDeferredRecord:
    return {
        "type": "write_deferred",
        "id": f"write-{seq}",
        "lane": "main",
        "seq": seq,
        "timestamp": seq,
        "runId": "run-1",
        "target": target or message_target("write-1", user_message("deferred write")),
    }


def recovery_slice(
    records: list[LaneRecord],
    entries: list[Entry] | None = None,
) -> RecordLogSlice:
    finished = {
        cast(OperationFinishedRecord, record)["runId"]
        for record in records
        if record["type"] == "operation_finished"
    }
    open_operations = [
        cast(OperationStartedRecord, record)
        for record in records
        if record["type"] == "operation_started" and record["id"] not in finished
    ]
    open_operations.sort(key=lambda record: int(record["seq"]), reverse=True)
    return {
        "lane": "main",
        "openOperations": open_operations,
        "records": records,
        "entries": entries or [],
    }


DEFAULTS: EffectiveLaneConfiguration = {
    "model": {"provider": "default-provider", "modelId": "default-model"},
    "thinkingLevel": "off",
    "activeToolNames": ["default-tool"],
}


def reduction_input(
    records: list[LaneRecord],
    own_entries: list[Entry] | None = None,
    **options: Any,
) -> LaneReductionInput:
    own = own_entries or []
    slice_data = recovery_slice(records, [*own, *(options.get("entries") or [])])
    return {
        **slice_data,
        "leafId": options.get("leafId", own[-1]["id"] if own else None),
        "ownEntries": own,
        "configurationEntries": options.get("configurationEntries", []),
        "defaults": options.get("defaults", DEFAULTS),
    }


def expect_corruption(input: RecordLogSlice, reason: RecordLogCorruptionReason) -> None:
    with pytest.raises(RecordLogCorruption) as excinfo:
        validate_record_log(input)
    assert excinfo.value.reason == reason


ASSISTANT_TOOLS_ENTRY = persisted_entry(
    message_target(
        "assistant-tools",
        assistant_message(
            [{"type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {}}],
            "toolUse",
        ),
    ),
    3,
)


def test_idle_lane_reduction() -> None:
    result: LaneReductionResult = reduce_lane_state(reduction_input([], leafId=None))
    assert result["laneState"]["operation"] is None
    assert result["effectiveConfiguration"] == DEFAULTS
    assert result["terminalFailure"] is None


def test_run_reduction_exposes_step_and_configuration() -> None:
    result: LaneReductionResult = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-1")],
            configurationEntries=[
                {
                    "type": "thinking_level_change",
                    "id": "t1",
                    "seq": 1,
                    "parentId": None,
                    "timestamp": 1,
                    "thinkingLevel": "high",
                }
            ],
        )
    )
    operation = result["laneState"]["operation"]
    assert operation is not None
    assert operation["id"] == "run-1"
    assert operation["step"]["kind"] == "assistant"
    assert result["effectiveConfiguration"]["thinkingLevel"] == "high"


def test_tool_batch_reduction() -> None:
    own = [
        ASSISTANT_TOOLS_ENTRY,
        persisted_entry(message_target("tool-result-1", tool_result_message()), 4),
    ]
    result: LaneReductionResult = reduce_lane_state(
        reduction_input(
            [
                run_started(1),
                tool_started(2),
            ],
            own,
        )
    )
    operation = result["laneState"]["operation"]
    assert operation is not None
    tool_batch = operation["toolBatch"]
    assert tool_batch is not None
    assert tool_batch["unresolved"] is False
    assert tool_batch["calls"][0]["toolCall"]["name"] == "tool-1"


def test_terminal_failure_from_error_step() -> None:
    error_entry = persisted_entry(
        message_target(
            "assistant-error",
            assistant_message([{"type": "text", "text": "boom"}], "error"),
        ),
        3,
    )
    result: LaneReductionResult = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-error")],
            [error_entry],
        )
    )
    assert result["terminalFailure"] is not None
    assert result["terminalFailure"]["source"] == "step"
    assert result["terminalFailure"]["entryId"] == "assistant-error"


CORRUPTION_CASES: list[tuple[str, RecordLogCorruptionReason, RecordLogSlice]] = [
    (
        "multiple open operations",
        "multiple_open_operations",
        recovery_slice([run_started(1), run_started(2, id="run-2")]),
    ),
    (
        "unknown operation",
        "unknown_operation",
        recovery_slice([abort_requested(1, "missing")]),
    ),
    (
        "record after finish",
        "record_after_finish",
        recovery_slice([run_started(1), operation_finished(2), abort_requested(3)]),
    ),
    (
        "non-consecutive attempt",
        "non_consecutive_attempt",
        recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "assistant", 1, "assistant-1"),
                attempt(3, "run-1", "assistant", 3, "assistant-2"),
            ]
        ),
    ),
    (
        "non-compaction carries reason",
        "invalid_compaction_reason",
        recovery_slice(
            [
                run_started(1),
                {
                    **attempt(2, "run-1", "assistant", 1, "assistant-1"),
                    "compactionReason": "manual",
                },  # type: ignore[arg-type]
            ]
        ),
    ),
    (
        "queue after abort",
        "queue_after_abort",
        recovery_slice([run_started(1), abort_requested(2), queue_enqueued(3)]),
    ),
    (
        "queue cancellation without enqueue",
        "invalid_queue_cancellation",
        recovery_slice([run_started(1), queue_cancelled(2)]),
    ),
    (
        "queue cancellation targets existing entry",
        "invalid_queue_cancellation",
        recovery_slice(
            [run_started(1), queue_enqueued(2), queue_cancelled(4)],
            [persisted_entry(message_target("queue-1", user_message("queued")), 3)],
        ),
    ),
    (
        "structural attempts disagree on result",
        "inconsistent_step",
        recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "compaction", 1, "compaction-1", "threshold"),
                attempt(3, "run-1", "compaction", 2, "compaction-2", "threshold"),
            ]
        ),
    ),
    (
        "tool call mismatch",
        "tool_call_mismatch",
        recovery_slice(
            [run_started(1), tool_started(2, toolCallId="wrong")],
            [ASSISTANT_TOOLS_ENTRY],
        ),
    ),
    (
        "duplicate tool invocation",
        "duplicate_tool_invocation",
        recovery_slice(
            [
                run_started(1),
                tool_started(2),
                tool_started(3),
            ],
            [
                ASSISTANT_TOOLS_ENTRY,
                persisted_entry(message_target("tool-result-1", tool_result_message()), 4),
            ],
        ),
    ),
    (
        "provisioned entry mismatch",
        "provisioned_entry_mismatch",
        recovery_slice(
            [run_started(1, initialMessages=[message_target("queue-1", user_message("expected"))])],
            [persisted_entry(message_target("queue-1", user_message("actual")), 2)],
        ),
    ),
    (
        "deferred assistant without handle",
        "invalid_deferred_handle",
        recovery_slice(
            [],
            [
                persisted_entry(
                    message_target(
                        "deferred-no-handle",
                        {
                            **assistant_message([{"type": "text", "text": "x"}], "deferred"),
                            "deferred": None,
                        },
                    ),
                    1,
                )
            ],
        ),
    ),
]


@pytest.mark.parametrize(
    ("name", "reason", "input"),
    CORRUPTION_CASES,
    ids=[case[0] for case in CORRUPTION_CASES],
)
def test_record_log_corruption(
    name: str, reason: RecordLogCorruptionReason, input: RecordLogSlice
) -> None:
    expect_corruption(input, reason)
