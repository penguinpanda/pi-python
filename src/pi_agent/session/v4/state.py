"""v4 Session 纯内存归约（对齐 TS `harness/session/state.ts`）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Literal, TypedDict, cast

from typing_extensions import NotRequired

from .types import (
    BranchBounds,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    OperationStartedRecord,
    RecordQuery,
    SessionError,
    SessionStats,
)


class EntryMutation(TypedDict):
    kind: Literal["entry"]
    lane: NotRequired[str]
    entry: Entry


class RecordMutation(TypedDict):
    kind: Literal["record"]
    record: LaneRecord


class LaneMutation(TypedDict):
    kind: Literal["lane"]
    seq: int
    lane: str
    leafId: str | None


class NameFactMutation(TypedDict):
    kind: Literal["fact"]
    seq: int
    fact: Literal["name"]
    name: str


class LabelFactMutation(TypedDict):
    kind: Literal["fact"]
    seq: int
    fact: Literal["label"]
    targetId: str
    label: str | None


SessionMutation = (
    EntryMutation | RecordMutation | LaneMutation | NameFactMutation | LabelFactMutation
)

EntryOrder = Literal["newestFirst", "oldestFirst"]


def _invalid_mutation(message: str) -> None:
    raise SessionError("invalid_entry", f"Invalid session mutation: {message}")


def _assert_valid_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def _assert_valid_cursor(after_seq: int | None) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


def _ordered(items: list[Any], order: EntryOrder | None) -> list[Any]:
    return list(items) if order == "oldestFirst" else list(reversed(items))


class SessionState:
    """v4 会话状态的纯内存实现：seq / lanes / entries / records / facts / stats。"""

    def __init__(self) -> None:
        self._sequence = 0
        self._used_ids: set[str] = set()
        self._entries: list[Entry] = []
        self._entries_by_id: dict[str, Entry] = {}
        self._records: list[LaneRecord] = []
        self._open_operations_by_lane: dict[str, dict[str, OperationStartedRecord]] = {}
        self._lanes: dict[str, str | None] = {"main": None}
        self._log: list[LogItem] = []
        self._stats: SessionStats = {
            "messageCount": 0,
            "cachedTokens": 0,
            "uncachedTokens": 0,
            "totalTokens": 0,
            "costTotal": 0.0,
        }
        self._name: str | None = None
        self._labels: dict[str, str] = {}

    @property
    def next_sequence(self) -> int:
        return self._sequence + 1

    def get_lanes(self) -> list[LanePointer]:
        return [{"lane": lane, "leafId": leaf_id} for lane, leaf_id in self._lanes.items()]

    def require_lane(self, lane: str) -> str | None:
        if lane not in self._lanes:
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        return self._lanes[lane]

    def validate_new_lane(self, lane: str) -> None:
        if lane in self._lanes:
            raise SessionError("already_exists", f"Lane already exists: {lane}")

    def validate_target(self, target_id: str | None) -> None:
        if target_id is not None and target_id not in self._entries_by_id:
            raise SessionError("not_found", f"Entry not found: {target_id}")

    def validate_unused_id(self, entry_id: str) -> None:
        if entry_id in self._used_ids:
            raise SessionError("already_exists", f"Session id already exists: {entry_id}")

    def apply_mutation(
        self,
        mutation: SessionMutation,
        invalid: Callable[[str], None] = _invalid_mutation,
    ) -> None:
        # mypy 对 TypedDict 判别联合的逐字段收窄不稳定，统一按运行时键访问。
        m: Any = mutation
        if m["kind"] == "entry":
            seq: int = m["entry"]["seq"]
        elif m["kind"] == "record":
            seq = m["record"]["seq"]
        else:
            seq = m["seq"]
        if seq != self._sequence + 1:
            invalid(f"has non-consecutive seq {seq}")

        kind = m["kind"]
        if kind == "entry":
            entry = m["entry"]
            if entry["id"] in self._used_ids:
                invalid(f"contains duplicate id {entry['id']}")
            lane = m.get("lane")
            if lane is not None:
                leaf_id = self._lanes.get(lane)
                if leaf_id is None and lane not in self._lanes:
                    invalid(f"references missing lane {lane}")
                if entry["parentId"] != leaf_id:
                    invalid("does not chain to the lane leaf")
            parent_id = entry["parentId"]
            if parent_id is not None and parent_id not in self._entries_by_id:
                invalid(f"references missing parent {parent_id}")
            self._sequence = seq
            self._used_ids.add(entry["id"])
            self._entries.append(entry)
            self._entries_by_id[entry["id"]] = entry
            if lane is not None:
                self._lanes[lane] = entry["id"]
            self._log.append({"kind": "entry", "seq": seq, "entry": entry})
            if entry["type"] == "message":
                self._stats["messageCount"] += 1
            return

        if kind == "record":
            record = m["record"]
            if record["lane"] not in self._lanes:
                invalid(f"references missing lane {record['lane']}")
            if record["id"] in self._used_ids:
                invalid(f"contains duplicate id {record['id']}")
            self._sequence = seq
            self._used_ids.add(record["id"])
            self._records.append(record)
            if record["type"] == "operation_started":
                self._open_operations_by_lane.setdefault(record["lane"], {})[record["id"]] = record
            elif record["type"] == "operation_finished":
                open_by_lane = self._open_operations_by_lane.get(record["lane"])
                if open_by_lane is not None:
                    open_by_lane.pop(record["runId"], None)
            self._log.append({"kind": "record", "seq": seq, "record": record})
            if record["type"] == "usage":
                usage = record["usage"]
                self._stats["cachedTokens"] += usage["cache_read"]
                self._stats["uncachedTokens"] += usage["input"] + usage["cache_write"]
                self._stats["totalTokens"] += usage["total_tokens"]
                self._stats["costTotal"] += usage["cost"]["total"]
            return

        if kind == "lane":
            if m["leafId"] is not None and m["leafId"] not in self._entries_by_id:
                invalid(f"references missing lane target {m['leafId']}")
            self._sequence = seq
            self._lanes[m["lane"]] = m["leafId"]
            self._log.append(
                {
                    "kind": "lane",
                    "seq": seq,
                    "lane": m["lane"],
                    "leafId": m["leafId"],
                }
            )
            return

        # fact
        if m["fact"] == "label" and m["targetId"] not in self._entries_by_id:
            invalid(f"references missing label target {m['targetId']}")
        self._sequence = seq
        if m["fact"] == "name":
            self._name = m["name"]
            self._log.append({"kind": "fact", "seq": seq, "fact": "name", "name": m["name"]})
        else:
            if m["label"] is None:
                self._labels.pop(m["targetId"], None)
            else:
                self._labels[m["targetId"]] = m["label"]
            self._log.append(
                {
                    "kind": "fact",
                    "seq": seq,
                    "fact": "label",
                    "targetId": m["targetId"],
                    "label": m["label"],
                }
            )

    def get_entry(self, entry_id: str) -> Entry | None:
        return self._entries_by_id.get(entry_id)

    def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        query = query or {}
        _assert_valid_limit(query.get("limit"))
        cursor = query.get("cursor")
        _assert_valid_cursor(cursor["afterSeq"] if cursor is not None else None)
        results: list[Entry] = []
        for entry in _ordered(self._entries, query.get("order")):
            if not self._matches_entry_query(entry, query):
                continue
            results.append(entry)
            if query.get("limit") is not None and len(results) >= query["limit"]:
                break
        return results

    def find_entries_on_branch(self, query: dict[str, Any]) -> list[Entry]:
        _assert_valid_limit(query.get("limit"))
        _assert_valid_cursor((query.get("cursor") or {}).get("afterSeq"))
        results: list[Entry] = []
        start = query.get("start")
        if not isinstance(start, str):
            raise SessionError("invalid_query", "branch query requires start")
        order = query.get("order")
        if order == "oldestFirst":
            walked = list(self._walk_to_root(start))[::-1]
            for entry in walked:
                reached_bound = entry["id"] == query.get("stopAtId") or entry["type"] == query.get(
                    "stopAtType"
                )
                if self._matches_entry_query(entry, cast(EntryQuery, query)):
                    results.append(entry)
                if reached_bound or (
                    query.get("limit") is not None and len(results) >= query["limit"]
                ):
                    break
        else:
            for entry in self._walk_to_root(start, cast(BranchBounds, query)):
                if self._matches_entry_query(entry, cast(EntryQuery, query)):
                    results.append(entry)
                if query.get("limit") is not None and len(results) >= query["limit"]:
                    break
        return results

    def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        query = query or {}
        _assert_valid_limit(query.get("limit"))
        _assert_valid_cursor(query.get("afterSeq"))
        results: list[LaneRecord] = []
        for record in _ordered(self._records, query.get("order")):
            if not self._matches_record_query(record, query):
                continue
            results.append(record)
            if query.get("limit") is not None and len(results) >= query["limit"]:
                break
        return results

    def find_open_operations(
        self, lane: str, options: dict[str, int] | None = None
    ) -> list[OperationStartedRecord]:
        options = options or {}
        _assert_valid_limit(options.get("limit"))
        open_operations = self._open_operations_by_lane.get(lane)
        operations = list(reversed(list(open_operations.values()))) if open_operations else []
        limit = options.get("limit")
        return operations if limit is None else operations[:limit]

    def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options or {}
        _assert_valid_limit(options.get("limit"))
        _assert_valid_cursor(options.get("afterSeq"))
        after_seq = options.get("afterSeq")
        results: list[LogItem] = []
        for item in self._log:
            if after_seq is not None and item["seq"] <= after_seq:
                continue
            results.append(item)
            if options.get("limit") is not None and len(results) >= options["limit"]:
                break
        return results

    def get_name(self) -> str | None:
        return self._name

    def get_label(self, entry_id: str) -> str | None:
        return self._labels.get(entry_id)

    def get_stats(self) -> SessionStats:
        return cast(SessionStats, dict(self._stats))

    def create_fork_mutations(self, options: ForkOptions) -> list[SessionMutation]:
        if options.get("scope") == "tree":
            copied_entries = self.find_entries({"order": "oldestFirst"})
            fork_lanes = self.get_lanes()
        else:
            selected_entry_id = options.get("entryId") or self.require_lane("main")
            target_id: str | None = None
            if selected_entry_id is not None:
                entry = self.get_entry(selected_entry_id)
                if entry is None or entry["type"] != "message":
                    raise SessionError(
                        "invalid_fork_target",
                        f"Fork target is not a message entry: {selected_entry_id}",
                    )
                position = options.get("position") or (
                    "at" if options.get("entryId") is None else "before"
                )
                target_id = entry["id"] if position == "at" else entry["parentId"]
            copied_entries = (
                []
                if target_id is None
                else self.find_entries_on_branch({"start": target_id, "order": "oldestFirst"})
            )
            fork_lanes = [{"lane": "main", "leafId": target_id}]

        mutations: list[SessionMutation] = []
        sequence = 1
        for source_entry in copied_entries:
            entry = deepcopy(source_entry)
            entry["seq"] = sequence
            mutations.append({"kind": "entry", "entry": entry})
            sequence += 1
        for pointer in fork_lanes:
            mutations.append(
                {
                    "kind": "lane",
                    "seq": sequence,
                    "lane": pointer["lane"],
                    "leafId": pointer["leafId"],
                }
            )
            sequence += 1
        if self._name is not None:
            mutations.append(
                {
                    "kind": "fact",
                    "seq": sequence,
                    "fact": "name",
                    "name": self._name,
                }
            )
            sequence += 1
        for entry in copied_entries:
            label = self._labels.get(entry["id"])
            if label is not None:
                mutations.append(
                    {
                        "kind": "fact",
                        "seq": sequence,
                        "fact": "label",
                        "targetId": entry["id"],
                        "label": label,
                    }
                )
                sequence += 1
        return mutations

    def _walk_to_root(self, start: str | None, bounds: BranchBounds | None = None) -> list[Entry]:
        if start is None:
            return []
        visited: set[str] = set()
        current = self._entries_by_id.get(start)
        if current is None:
            raise SessionError("not_found", f"Entry not found: {start}")
        result: list[Entry] = []
        while current is not None:
            if current["id"] in visited:
                raise SessionError(
                    "invalid_entry", f"Session branch contains a cycle at {current['id']}"
                )
            visited.add(current["id"])
            result.append(current)
            bounds_dict = bounds or {}
            if (
                current["id"] == bounds_dict.get("stopAtId")
                or current["type"] == bounds_dict.get("stopAtType")
                or current["parentId"] is None
            ):
                break
            parent_id = current["parentId"]
            parent = self._entries_by_id.get(parent_id) if parent_id is not None else None
            if parent is None:
                raise SessionError("invalid_entry", f"Entry not found: {parent_id}")
            current = parent
        return result

    @staticmethod
    def _matches_entry_query(entry: Entry, query: EntryQuery) -> bool:
        cursor = query.get("cursor")
        if cursor is not None:
            after_seq = cursor["afterSeq"]
            matches_cursor = (
                entry["seq"] > after_seq
                if query.get("order") == "oldestFirst"
                else entry["seq"] < after_seq
            )
        else:
            matches_cursor = True
        return (
            (query.get("type") is None or entry["type"] == query["type"])
            and (
                query.get("customType") is None
                or (
                    entry["type"] == "custom"
                    and cast(Any, entry)["customType"] == query["customType"]
                )
            )
            and matches_cursor
        )

    @staticmethod
    def _matches_record_query(record: LaneRecord, query: RecordQuery) -> bool:
        run_id = query.get("runId")
        matches_run_id: bool
        if run_id is None:
            matches_run_id = True
        elif record["type"] == "operation_started":
            matches_run_id = record["id"] == run_id
        else:
            matches_run_id = cast(Any, record).get("runId") == run_id
        operation_kind = query.get("operationKind")
        matches_operation_kind = operation_kind is None or (
            record["type"] == "operation_started"
            and cast(Any, record)["intent"]["kind"] == operation_kind
        )
        after_seq = query.get("afterSeq")
        return (
            (query.get("lane") is None or record["lane"] == query["lane"])
            and (query.get("type") is None or record["type"] == query["type"])
            and matches_run_id
            and matches_operation_kind
            and (after_seq is None or record["seq"] > after_seq)
        )


__all__ = [
    "SessionState",
    "SessionMutation",
    "EntryMutation",
    "RecordMutation",
    "LaneMutation",
    "NameFactMutation",
    "LabelFactMutation",
]
