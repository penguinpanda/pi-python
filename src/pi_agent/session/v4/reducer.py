"""Record log 校验与 lane 状态归约（对齐 TS `harness/reducer.ts`）。

只读取有界恢复切片，不做任何持久化写入；检测到单写者协议不可能产生的
损坏状态时抛出 `RecordLogCorruption`。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict, cast

from typing_extensions import NotRequired

from .types import (
    Entry,
    LaneRecord,
    OperationFinishedRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    StepAttemptRecord,
    ToolStartedRecord,
    WriteDeferredRecord,
)

RecordLogCorruptionReason = Literal[
    "multiple_open_operations",
    "unknown_operation",
    "record_after_finish",
    "non_consecutive_attempt",
    "invalid_compaction_reason",
    "queue_after_abort",
    "invalid_queue_cancellation",
    "inconsistent_step",
    "tool_call_mismatch",
    "duplicate_tool_invocation",
    "provisioned_entry_mismatch",
    "invalid_deferred_handle",
]


class RecordLogCorruption(Exception):
    def __init__(self, reason: RecordLogCorruptionReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class RecordLogSlice(TypedDict):
    lane: str
    openOperations: list[OperationStartedRecord]
    records: list[LaneRecord]
    entries: list[Entry]


class EffectiveLaneConfiguration(TypedDict):
    model: dict[str, str]
    thinkingLevel: str
    activeToolNames: list[str]


class TerminalFailureState(TypedDict):
    entryId: str
    source: Literal["step", "deferred_fetch"]
    message: dict[str, Any]


class ToolCallState(TypedDict):
    toolIndex: int
    toolCall: dict[str, Any]
    started: NotRequired[ToolStartedRecord]
    resultExists: bool
    terminate: NotRequired[bool]


class ToolBatchState(TypedDict):
    assistantEntryId: str
    calls: list[ToolCallState]
    truncated: bool
    unresolved: bool


class LaneOperationState(TypedDict):
    id: str
    kind: str
    intent: dict[str, Any]
    aborting: bool
    step: NotRequired[dict[str, Any]]
    toolBatch: ToolBatchState | None
    missingInitialMessages: list[ProvisionedEntry]
    pendingSteer: list[ProvisionedEntry]
    pendingFollowUp: list[ProvisionedEntry]
    pendingWrites: list[ProvisionedEntry]
    deferred: NotRequired[dict[str, Any]]
    overflowRecoveryUsed: bool
    newestOwn: NotRequired[dict[str, Any]]
    targets: dict[str, bool]


class LaneState(TypedDict):
    lane: str
    leafId: str | None
    operation: LaneOperationState | None
    pendingNextRun: list[ProvisionedEntry]


class LaneReductionInput(RecordLogSlice):
    leafId: str | None
    ownEntries: list[Entry]
    configurationEntries: list[Entry]
    defaults: EffectiveLaneConfiguration


class LaneReductionResult(TypedDict):
    laneState: LaneState
    effectiveConfiguration: EffectiveLaneConfiguration
    terminalFailure: TerminalFailureState | None


class _AttemptSeries(TypedDict):
    record: StepAttemptRecord


def _corrupt(reason: RecordLogCorruptionReason, message: str) -> None:
    raise RecordLogCorruption(reason, message)


def _has_run_id(record: LaneRecord) -> bool:
    value = cast(dict[str, Any], record).get("runId")
    return isinstance(value, str)


def _matches_provisioned_entry(entry: Entry, target: ProvisionedEntry) -> bool:
    payload = {
        key: value for key, value in entry.items() if key not in ("parentId", "seq", "timestamp")
    }
    return payload == target


def _validate_exact_provisioned_entry(
    entries_by_id: dict[str, Entry],
    target: ProvisionedEntry,
) -> None:
    entry = entries_by_id.get(target["id"])
    if entry is not None and not _matches_provisioned_entry(entry, target):
        _corrupt(
            "provisioned_entry_mismatch",
            f"Provisioned entry {target['id']} exists with content different from its intent",
        )


def _validate_result_entry(
    entries_by_id: dict[str, Entry],
    result_entry_id: str,
    matches: Any,
    description: str,
) -> None:
    entry = entries_by_id.get(result_entry_id)
    if entry is not None and not matches(entry):
        _corrupt(
            "provisioned_entry_mismatch",
            f"Provisioned {description} entry {result_entry_id} exists with different content",
        )


def _validate_attempt_reason(record: StepAttemptRecord) -> None:
    reason = cast(dict[str, Any], record).get("compactionReason")
    if record["step"] == "compaction":
        if reason not in ("manual", "threshold", "overflow"):
            _corrupt(
                "invalid_compaction_reason",
                f"Compaction attempt {record['id']} has no valid compaction reason",
            )
    elif reason is not None:
        _corrupt(
            "invalid_compaction_reason",
            f"{record['step']} attempt {record['id']} has a compaction reason",
        )


def _validate_attempt_sequence(
    record: StepAttemptRecord,
    previous: _AttemptSeries | None,
    entries_by_id: dict[str, Entry],
) -> None:
    previous_record = previous["record"] if previous is not None else None
    previous_result = (
        entries_by_id.get(previous_record["resultEntryId"]) if previous_record is not None else None
    )
    continues_series = (
        previous_record is not None
        and previous_record["step"] == record["step"]
        and (
            previous_result is None
            or int(cast(dict[str, Any], previous_result)["seq"]) >= int(record["seq"])
        )
    )
    expected_attempt = (
        (previous_record["attempt"] + 1 if continues_series else 1) if previous_record else 1
    )
    if record["attempt"] != expected_attempt:
        _corrupt(
            "non_consecutive_attempt",
            f"{record['step']} attempt {record['id']} is {record['attempt']}; expected {expected_attempt}",
        )
    if not continues_series or record["step"] == "assistant" or previous_record is None:
        return
    if record["resultEntryId"] != previous_record["resultEntryId"]:
        _corrupt(
            "inconsistent_step", f"{record['step']} attempts disagree on their result entry id"
        )
    if cast(dict[str, Any], record).get("compactionReason") != cast(
        dict[str, Any], previous_record
    ).get("compactionReason"):
        _corrupt(
            "inconsistent_step", f"{record['step']} attempts disagree on their compaction reason"
        )


def _validate_attempt_result(entries_by_id: dict[str, Entry], record: StepAttemptRecord) -> None:
    step = record["step"]
    if step == "assistant":

        def _assistant(entry: Entry) -> bool:
            return (
                entry["type"] == "message"
                and cast(dict[str, Any], entry["message"]).get("role") == "assistant"
            )

        _validate_result_entry(
            entries_by_id, record["resultEntryId"], _assistant, "assistant result"
        )
    elif step == "compaction":
        _validate_result_entry(
            entries_by_id,
            record["resultEntryId"],
            lambda entry: entry["type"] == "compaction",
            "compaction result",
        )
    else:
        _validate_result_entry(
            entries_by_id,
            record["resultEntryId"],
            lambda entry: entry["type"] == "branch_summary",
            "branch-summary result",
        )


def _validate_tool_start(
    record: ToolStartedRecord,
    entries_by_id: dict[str, Entry],
    invocations: set[str],
) -> None:
    invocation = f"{record['assistantEntryId']}\x00{record['toolIndex']}"
    if invocation in invocations:
        _corrupt(
            "duplicate_tool_invocation",
            f"Tool invocation {record['assistantEntryId']}:{record['toolIndex']} is duplicated",
        )
    invocations.add(invocation)

    assistant_entry = entries_by_id.get(record["assistantEntryId"])
    if (
        assistant_entry is None
        or assistant_entry["type"] != "message"
        or cast(dict[str, Any], assistant_entry["message"]).get("role") != "assistant"
    ):
        _corrupt(
            "tool_call_mismatch", f"Tool start {record['id']} does not reference an assistant entry"
        )
    message = cast(dict[str, Any], cast(dict[str, Any], assistant_entry)["message"])
    tool_calls = [
        block for block in message.get("content") or [] if block.get("type") == "toolCall"
    ]
    tool_call = tool_calls[record["toolIndex"]] if record["toolIndex"] < len(tool_calls) else None
    if (
        tool_call is None
        or tool_call.get("id") != record["toolCallId"]
        or tool_call.get("name") != record["toolName"]
    ):
        _corrupt(
            "tool_call_mismatch",
            f"Tool start {record['id']} does not match its assistant tool-call ordinal",
        )

    def _tool_result(entry: Entry) -> bool:
        if entry["type"] != "message":
            return False
        result_message = cast(dict[str, Any], entry["message"])
        return (
            result_message.get("role") == "toolResult"
            and result_message.get("tool_call_id") == record["toolCallId"]
            and result_message.get("tool_name") == record["toolName"]
        )

    _validate_result_entry(entries_by_id, record["resultEntryId"], _tool_result, "tool result")


def _validate_deferred_handles(entries: list[Entry]) -> None:
    for entry in entries:
        if entry["type"] != "message":
            continue
        message = cast(dict[str, Any], entry["message"])
        if (
            message.get("role") == "assistant"
            and message.get("stop_reason") == "deferred"
            and not message.get("deferred")
        ):
            _corrupt(
                "invalid_deferred_handle",
                f"Deferred assistant entry {entry['id']} does not carry a handle",
            )


def _validate_operation_result(
    entries_by_id: dict[str, Entry],
    record: OperationStartedRecord,
) -> None:
    intent = cast(dict[str, Any], record["intent"])
    kind = intent["kind"]
    if kind == "run":
        for target in intent["initialMessages"]:
            _validate_exact_provisioned_entry(entries_by_id, target)
    elif kind == "compaction":
        _validate_result_entry(
            entries_by_id,
            intent["resultEntryId"],
            lambda entry: entry["type"] == "compaction",
            "manual compaction",
        )
    elif intent.get("summaryEntryId"):
        _validate_result_entry(
            entries_by_id,
            intent["summaryEntryId"],
            lambda entry: entry["type"] == "branch_summary",
            "navigation summary",
        )


def validate_record_log(input: RecordLogSlice) -> None:
    """校验有界恢复切片；不读取或修改会话状态。"""
    if len(input["openOperations"]) > 1:
        _corrupt(
            "multiple_open_operations", f"Lane {input['lane']} has at least two open operations"
        )

    entries_by_id = {entry["id"]: entry for entry in input["entries"]}
    _validate_deferred_handles(input["entries"])
    starts: dict[str, OperationStartedRecord] = {}
    finished_at: dict[str, int] = {}
    aborted_at: dict[str, int] = {}
    queue_enqueues: dict[str, QueueEnqueuedRecord] = {}
    latest_attempt: dict[str, _AttemptSeries] = {}
    tool_invocations: set[str] = set()
    records = sorted(input["records"], key=lambda record: int(cast(dict[str, Any], record)["seq"]))

    for record in records:
        if record["type"] == "operation_started":
            starts[record["id"]] = record
            _validate_operation_result(entries_by_id, record)
            continue

        if _has_run_id(record):
            run_id = cast(dict[str, Any], record)["runId"]
            if run_id not in starts:
                _corrupt(
                    "unknown_operation",
                    f"Record {record['id']} references unknown operation {run_id}",
                )
            finish_seq = finished_at.get(run_id)
            if finish_seq is not None and int(cast(dict[str, Any], record)["seq"]) > finish_seq:
                _corrupt(
                    "record_after_finish",
                    f"Record {record['id']} follows the finish of operation {run_id}",
                )

        record_type = record["type"]
        if record_type == "operation_finished":
            finished_at[cast(OperationFinishedRecord, record)["runId"]] = cast(
                dict[str, Any], record
            )["seq"]
        elif record_type == "abort_requested":
            aborted_at[cast(dict[str, Any], record)["runId"]] = cast(dict[str, Any], record)["seq"]
        elif record_type == "step_attempt":
            attempt_record = cast(StepAttemptRecord, record)
            _validate_attempt_reason(attempt_record)
            _validate_attempt_sequence(
                attempt_record,
                latest_attempt.get(cast(dict[str, Any], record)["runId"]),
                entries_by_id,
            )
            _validate_attempt_result(entries_by_id, attempt_record)
            latest_attempt[cast(dict[str, Any], record)["runId"]] = {"record": attempt_record}
        elif record_type == "tool_started":
            _validate_tool_start(cast(ToolStartedRecord, record), entries_by_id, tool_invocations)
        elif record_type == "queue_enqueued":
            enqueue = cast(QueueEnqueuedRecord, record)
            if (
                enqueue["queue"] != "nextRun"
                and aborted_at.get(cast(dict[str, Any], record).get("runId", "")) is not None
                and int(cast(dict[str, Any], record)["seq"])
                > aborted_at[cast(dict[str, Any], record)["runId"]]
            ):
                _corrupt(
                    "queue_after_abort",
                    f"{enqueue['queue']} item {enqueue['target']['id']} was enqueued after abort",
                )
            queue_enqueues[enqueue["target"]["id"]] = enqueue
            _validate_exact_provisioned_entry(entries_by_id, enqueue["target"])
        elif record_type == "queue_cancelled":
            cancelled = cast(QueueCancelledRecord, record)
            matching_enqueue = queue_enqueues.get(cancelled["entryId"])
            if (
                matching_enqueue is None
                or int(cast(dict[str, Any], matching_enqueue)["seq"]) >= int(cancelled["seq"])
                or cast(dict[str, Any], matching_enqueue).get("runId")
                != cast(dict[str, Any], cancelled).get("runId")
                or cancelled["entryId"] in entries_by_id
            ):
                _corrupt(
                    "invalid_queue_cancellation",
                    f"Queue cancellation {cancelled['id']} has no pending matching enqueue",
                )
        elif record_type == "write_deferred":
            _validate_exact_provisioned_entry(
                entries_by_id, cast(WriteDeferredRecord, record)["target"]
            )
        elif record_type == "usage":
            continue


def _by_sequence(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda value: int(cast(dict[str, Any], value)["seq"]))


def _derive_effective_configuration(input: LaneReductionInput) -> EffectiveLaneConfiguration:
    configuration = deepcopy(input["defaults"])
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input["configurationEntries"], *input["ownEntries"]]:
        entries_by_id[entry["id"]] = entry

    for entry in _by_sequence(list(entries_by_id.values())):
        entry_type = entry["type"]
        if entry_type == "model_change":
            configuration["model"] = {
                "provider": cast(dict[str, Any], entry)["provider"],
                "modelId": cast(dict[str, Any], entry)["modelId"],
            }
        elif entry_type == "thinking_level_change":
            configuration["thinkingLevel"] = cast(dict[str, Any], entry)["thinkingLevel"]
        elif entry_type == "active_tools_change":
            configuration["activeToolNames"] = list(cast(dict[str, Any], entry)["activeToolNames"])
        elif entry_type == "message":
            message = cast(dict[str, Any], entry["message"])
            if message.get("role") == "assistant":
                configuration["model"] = {
                    "provider": message.get("provider", ""),
                    "modelId": message.get("model", ""),
                }
    return configuration


def _derive_newest_own(entry: Entry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    if entry["type"] != "message":
        return {"entryId": entry["id"], "type": entry["type"]}
    message = cast(dict[str, Any], entry["message"])
    if message.get("role") != "assistant":
        return {"entryId": entry["id"], "type": entry["type"], "role": message.get("role")}
    result: dict[str, Any] = {
        "entryId": entry["id"],
        "type": entry["type"],
        "role": "assistant",
        "stopReason": message.get("stop_reason"),
    }
    return result


def _derive_tool_batch(
    operation_id: str,
    records: list[LaneRecord],
    own_entries: list[Entry],
    entries_by_id: dict[str, Entry],
    deferred_write_ids: set[str],
) -> ToolBatchState | None:
    assistant_entry: Entry | None = None
    for entry in reversed(own_entries):
        if entry["type"] != "message":
            continue
        message = cast(dict[str, Any], entry["message"])
        if message.get("role") == "assistant" and any(
            block.get("type") == "toolCall" for block in message.get("content") or []
        ):
            assistant_entry = entry
            break
    if assistant_entry is None or assistant_entry["type"] != "message":
        return None
    message = cast(dict[str, Any], assistant_entry["message"])
    tool_calls = [
        block for block in message.get("content") or [] if block.get("type") == "toolCall"
    ]
    starts: dict[int, ToolStartedRecord] = {}
    for record in records:
        if (
            record["type"] == "tool_started"
            and cast(dict[str, Any], record).get("runId") == operation_id
            and cast(dict[str, Any], record).get("assistantEntryId") == assistant_entry["id"]
        ):
            starts[cast(dict[str, Any], record)["toolIndex"]] = cast(ToolStartedRecord, record)

    calls: list[ToolCallState] = []
    for tool_index, tool_call in enumerate(tool_calls):
        started = starts.get(tool_index)
        started_result = (
            entries_by_id.get(started["resultEntryId"]) if started is not None else None
        )
        blocked_result: Entry | None = None
        for entry in own_entries:
            if (
                entry["id"] in deferred_write_ids
                or entry["type"] != "message"
                or int(entry["seq"]) <= int(assistant_entry["seq"])
            ):
                continue
            result_message = cast(dict[str, Any], entry["message"])
            if result_message.get("role") == "toolResult" and result_message.get(
                "tool_call_id"
            ) == tool_call.get("id"):
                blocked_result = entry
                break
        result = started_result or blocked_result
        call: ToolCallState = {
            "toolIndex": tool_index,
            "toolCall": deepcopy(tool_call),
            "resultExists": result is not None,
        }
        if started is not None:
            call["started"] = deepcopy(started)
        if result is not None and result["type"] == "message" and result.get("terminate") is True:
            call["terminate"] = True
        calls.append(call)

    return {
        "assistantEntryId": assistant_entry["id"],
        "calls": calls,
        "truncated": message.get("stop_reason") == "length",
        "unresolved": any(not call["resultExists"] for call in calls),
    }


def reduce_lane_state(input: LaneReductionInput) -> LaneReductionResult:
    """从有界恢复输入重建单 lane 编排状态。"""
    validate_record_log(input)

    records = cast(list[LaneRecord], _by_sequence(input["records"]))
    own_entries = cast(list[Entry], _by_sequence(input["ownEntries"]))
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input["entries"], *own_entries]:
        entries_by_id[entry["id"]] = entry
    cancelled_queue_ids = {
        cast(QueueCancelledRecord, record)["entryId"]
        for record in records
        if record["type"] == "queue_cancelled"
    }
    pending_queue_records = [
        cast(QueueEnqueuedRecord, record)
        for record in records
        if record["type"] == "queue_enqueued"
        and record["target"]["id"] not in entries_by_id
        and record["target"]["id"] not in cancelled_queue_ids
    ]
    started = input["openOperations"][0] if input["openOperations"] else None
    captured_initial_message_ids = {
        target["id"]
        for target in (
            started["intent"]["initialMessages"]
            if started and started["intent"]["kind"] == "run"
            else []
        )
    }
    pending_next_run = [
        deepcopy(record["target"])
        for record in pending_queue_records
        if record["queue"] == "nextRun"
        and record["target"]["id"] not in captured_initial_message_ids
    ]
    effective_configuration = _derive_effective_configuration(input)

    if started is None:
        return {
            "laneState": {
                "lane": input["lane"],
                "leafId": input["leafId"],
                "operation": None,
                "pendingNextRun": pending_next_run,
            },
            "effectiveConfiguration": effective_configuration,
            "terminalFailure": None,
        }

    operation_records = [
        record
        for record in records
        if record["type"] == "operation_started"
        and record["id"] == started["id"]
        or _has_run_id(record)
        and cast(dict[str, Any], record).get("runId") == started["id"]
    ]
    aborting = any(record["type"] == "abort_requested" for record in operation_records)
    pending_steer = (
        []
        if aborting
        else [
            deepcopy(record["target"])
            for record in pending_queue_records
            if record["queue"] == "steer"
            and cast(dict[str, Any], record).get("runId") == started["id"]
        ]
    )
    pending_follow_up = (
        []
        if aborting
        else [
            deepcopy(record["target"])
            for record in pending_queue_records
            if record["queue"] == "followUp"
            and cast(dict[str, Any], record).get("runId") == started["id"]
        ]
    )
    pending_writes = [
        deepcopy(cast(WriteDeferredRecord, record)["target"])
        for record in operation_records
        if record["type"] == "write_deferred" and record["target"]["id"] not in entries_by_id
    ]
    missing_initial_messages = (
        [
            deepcopy(target)
            for target in started["intent"]["initialMessages"]
            if target["id"] not in entries_by_id
        ]
        if started["intent"]["kind"] == "run"
        else []
    )

    step_records = [
        cast(StepAttemptRecord, record)
        for record in operation_records
        if record["type"] == "step_attempt"
    ]
    newest_attempt = step_records[-1] if step_records else None
    step: dict[str, Any] | None = None
    if newest_attempt is not None and newest_attempt["resultEntryId"] not in entries_by_id:
        step = {
            "kind": newest_attempt["step"],
            "attempts": newest_attempt["attempt"],
            "resultEntryId": newest_attempt["resultEntryId"],
        }
        if newest_attempt["step"] == "compaction":
            step["compactionReason"] = cast(dict[str, Any], newest_attempt).get("compactionReason")

    consumed_input_ids: set[str] = set()
    if started["intent"]["kind"] == "run":
        consumed_input_ids.update(target["id"] for target in started["intent"]["initialMessages"])
    for record in operation_records:
        if record["type"] == "queue_enqueued" and record["queue"] != "nextRun":
            consumed_input_ids.add(record["target"]["id"])
    newest_consumed_input_sequence = float("-inf")
    for entry_id in consumed_input_ids:
        consumed_entry = entries_by_id.get(entry_id)
        if consumed_entry is not None and consumed_entry["type"] == "message":
            newest_consumed_input_sequence = max(
                newest_consumed_input_sequence, int(consumed_entry["seq"])
            )
    overflow_recovery_used = any(
        record["type"] == "step_attempt"
        and record["step"] == "compaction"
        and cast(dict[str, Any], record).get("compactionReason") == "overflow"
        and int(cast(dict[str, Any], record)["seq"]) > newest_consumed_input_sequence
        for record in operation_records
    )

    newest_own_entry = own_entries[-1] if own_entries else None
    newest_own = _derive_newest_own(newest_own_entry)
    deferred = None
    if (
        newest_own_entry is not None
        and newest_own_entry["type"] == "message"
        and cast(dict[str, Any], newest_own_entry["message"]).get("role") == "assistant"
        and cast(dict[str, Any], newest_own_entry["message"]).get("stop_reason") == "deferred"
    ):
        handle = cast(dict[str, Any], newest_own_entry["message"]).get("deferred")
        if handle is not None:
            deferred = deepcopy(handle)
    targets: dict[str, bool] = {}
    if started["intent"]["kind"] == "compaction":
        targets["result"] = started["intent"]["resultEntryId"] in entries_by_id
    elif started["intent"]["kind"] == "navigation" and started["intent"].get("summaryEntryId"):
        targets["summary"] = started["intent"]["summaryEntryId"] in entries_by_id

    deferred_write_ids = {
        cast(WriteDeferredRecord, record)["target"]["id"]
        for record in operation_records
        if record["type"] == "write_deferred"
    }
    terminal_failure: TerminalFailureState | None = None
    if (
        newest_own_entry is not None
        and newest_own_entry["type"] == "message"
        and cast(dict[str, Any], newest_own_entry["message"]).get("role") == "assistant"
        and cast(dict[str, Any], newest_own_entry["message"]).get("stop_reason") == "error"
        and newest_own_entry["id"] not in deferred_write_ids
    ):
        produced_by_step = any(
            record["type"] == "step_attempt" and record["resultEntryId"] == newest_own_entry["id"]
            for record in operation_records
        )
        previous_own_entry = own_entries[-2] if len(own_entries) >= 2 else None
        produced_by_deferred_fetch = any(
            record["type"] == "usage"
            and cast(dict[str, Any], record).get("cause") == "deferred_fetch"
            and cast(dict[str, Any], record).get("entryId") == newest_own_entry["id"]
            for record in operation_records
        ) or (
            previous_own_entry is not None
            and previous_own_entry["type"] == "message"
            and cast(dict[str, Any], previous_own_entry["message"]).get("role") == "assistant"
            and cast(dict[str, Any], previous_own_entry["message"]).get("stop_reason") == "deferred"
        )
        if produced_by_step or produced_by_deferred_fetch:
            terminal_failure = {
                "entryId": newest_own_entry["id"],
                "source": "step" if produced_by_step else "deferred_fetch",
                "message": deepcopy(cast(dict[str, Any], newest_own_entry["message"])),
            }

    operation: LaneOperationState = {
        "id": started["id"],
        "kind": started["intent"]["kind"],
        "intent": deepcopy(cast(dict[str, Any], started["intent"])),
        "aborting": aborting,
        "toolBatch": _derive_tool_batch(
            started["id"],
            operation_records,
            own_entries,
            entries_by_id,
            deferred_write_ids,
        ),
        "missingInitialMessages": missing_initial_messages,
        "pendingSteer": pending_steer,
        "pendingFollowUp": pending_follow_up,
        "pendingWrites": pending_writes,
        "overflowRecoveryUsed": overflow_recovery_used,
        "targets": targets,
    }
    if step is not None:
        operation["step"] = step
    if deferred is not None:
        operation["deferred"] = deferred
    if newest_own is not None:
        operation["newestOwn"] = newest_own

    return {
        "laneState": {
            "lane": input["lane"],
            "leafId": input["leafId"],
            "operation": operation,
            "pendingNextRun": pending_next_run,
        },
        "effectiveConfiguration": effective_configuration,
        "terminalFailure": terminal_failure,
    }


__all__ = [
    "RecordLogCorruption",
    "RecordLogCorruptionReason",
    "RecordLogSlice",
    "EffectiveLaneConfiguration",
    "TerminalFailureState",
    "ToolBatchState",
    "LaneState",
    "LaneReductionInput",
    "LaneReductionResult",
    "validate_record_log",
    "reduce_lane_state",
]
