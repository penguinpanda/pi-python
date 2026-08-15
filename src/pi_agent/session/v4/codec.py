"""JSONL v4 codec（对齐 TS `harness/session/jsonl/codec.ts`）。"""

from __future__ import annotations

import json
from typing import Any, cast

from .jsonl_types import JsonlSessionMetadata, JsonlV4Header
from .state import SessionMutation
from .types import Entry, LaneRecord, SessionError


ENTRY_TYPES = frozenset(
    {
        "message",
        "model_change",
        "thinking_level_change",
        "active_tools_change",
        "compaction",
        "branch_summary",
        "custom",
    }
)
RECORD_TYPES = frozenset(
    {
        "operation_started",
        "abort_requested",
        "operation_finished",
        "step_attempt",
        "tool_started",
        "queue_enqueued",
        "queue_cancelled",
        "write_deferred",
        "usage",
    }
)
OPERATION_KINDS = frozenset({"run", "compaction", "navigation"})


def invalid_file(
    path: str,
    line: int,
    message: str,
    cause: BaseException | None = None,
) -> SessionError:
    return SessionError(
        "invalid_entry",
        f"Invalid JSONL v4 session {path}: line {line} {message}",
        cause,
    )


def _parse_object(line: str, path: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise invalid_file(path, line_number, "is not valid JSON", error) from error
    if not isinstance(value, dict):
        raise invalid_file(path, line_number, "is not a JSON object")
    return value


def _require_string(value: Any, path: str, line: int, field: str) -> str:
    if not isinstance(value, str):
        raise invalid_file(path, line, f"has invalid {field}")
    return value


def _is_safe_integer(value: int) -> bool:
    return -(2**53) <= value <= 2**53


def _require_sequence(value: Any, path: str, line: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or not _is_safe_integer(value)
    ):
        raise invalid_file(path, line, "has invalid seq")
    return value


def _require_timestamp(value: Any, path: str, line: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or not _is_safe_integer(value)
    ):
        raise invalid_file(path, line, "has invalid timestamp")
    return value


def _require_nullable_id(value: Any, path: str, line: int, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise invalid_file(path, line, f"has invalid {field}")
    return value


# ---------------------------------------------------------------------------
# 消息 / usage 线协议转换（内存 snake_case ↔ JSONL camelCase）
# ---------------------------------------------------------------------------

_MESSAGE_KEY_TO_WIRE = {
    "stop_reason": "stopReason",
    "tool_call_id": "toolCallId",
    "tool_name": "toolName",
    "added_tool_names": "addedToolNames",
    "error_message": "errorMessage",
    "raw_arguments": "rawArguments",
    "text_signature": "textSignature",
    "thinking_signature": "thinkingSignature",
    "mime_type": "mimeType",
}
_MESSAGE_KEY_FROM_WIRE = {value: key for key, value in _MESSAGE_KEY_TO_WIRE.items()}
_USAGE_KEY_TO_WIRE = {
    "cache_read": "cacheRead",
    "cache_write": "cacheWrite",
    "total_tokens": "totalTokens",
    "cache_write_1h": "cacheWrite1h",
}
_USAGE_KEY_FROM_WIRE = {value: key for key, value in _USAGE_KEY_TO_WIRE.items()}
_COST_KEY_TO_WIRE = {"cache_read": "cacheRead", "cache_write": "cacheWrite"}
_COST_KEY_FROM_WIRE = {value: key for key, value in _COST_KEY_TO_WIRE.items()}


def _transform_mapping(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            mapping.get(str(key), str(key)): _transform_mapping(item, mapping)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_transform_mapping(item, mapping) for item in value]
    return value


def _message_to_wire(message: Any) -> Any:
    return _transform_mapping(message, _MESSAGE_KEY_TO_WIRE)


def _message_from_wire(message: Any) -> Any:
    return _transform_mapping(message, _MESSAGE_KEY_FROM_WIRE)


def _usage_to_wire(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return usage
    result = {
        _USAGE_KEY_TO_WIRE.get(str(key), str(key)): _usage_to_wire(item)
        for key, item in usage.items()
    }
    cost = result.get("cost")
    if isinstance(cost, dict):
        result["cost"] = {
            _COST_KEY_TO_WIRE.get(str(key), str(key)): item for key, item in cost.items()
        }
    return result


def _usage_from_wire(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return usage
    result = {
        _USAGE_KEY_FROM_WIRE.get(str(key), str(key)): _usage_from_wire(item)
        for key, item in usage.items()
    }
    cost = result.get("cost")
    if isinstance(cost, dict):
        result["cost"] = {
            _COST_KEY_FROM_WIRE.get(str(key), str(key)): item for key, item in cost.items()
        }
    return result


def _entry_to_wire(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    result = dict(entry)
    if result.get("type") == "message" and isinstance(result.get("message"), dict):
        message = _message_to_wire(result["message"])
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            message["usage"] = _usage_to_wire(message["usage"])
        result["message"] = message
    if result.get("type") == "compaction":
        if isinstance(result.get("retainedTail"), list):
            result["retainedTail"] = [_message_to_wire(m) for m in result["retainedTail"]]
        if "usage" in result:
            result["usage"] = _usage_to_wire(result["usage"])
    if result.get("type") == "branch_summary" and "usage" in result:
        result["usage"] = _usage_to_wire(result["usage"])
    return result


def _entry_from_wire(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    result = dict(entry)
    if result.get("type") == "message" and isinstance(result.get("message"), dict):
        message = _message_from_wire(result["message"])
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            message["usage"] = _usage_from_wire(message["usage"])
        result["message"] = message
    if result.get("type") == "compaction":
        if isinstance(result.get("retainedTail"), list):
            result["retainedTail"] = [_message_from_wire(m) for m in result["retainedTail"]]
        if "usage" in result:
            result["usage"] = _usage_from_wire(result["usage"])
    if result.get("type") == "branch_summary" and "usage" in result:
        result["usage"] = _usage_from_wire(result["usage"])
    return result


def parse_header(line: str, path: str) -> JsonlV4Header:
    """解析首行 header（对齐 TS parseHeader）。"""
    value = _parse_object(line, path, 1)
    if value.get("kind") != "header":
        raise invalid_file(path, 1, "is not a header")
    if value.get("version") != 4:
        raise invalid_file(path, 1, "has unsupported session version")
    parent_session_id = value.get("parentSessionId")
    if parent_session_id is not None and not isinstance(parent_session_id, str):
        raise invalid_file(path, 1, "has invalid parentSessionId")
    legacy_parent_path = value.get("legacyParentSessionPath")
    if legacy_parent_path is not None and not isinstance(legacy_parent_path, str):
        raise invalid_file(path, 1, "has invalid legacyParentSessionPath")
    if parent_session_id is not None and legacy_parent_path is not None:
        raise invalid_file(path, 1, "has both parentSessionId and legacyParentSessionPath")
    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise invalid_file(path, 1, "has invalid metadata")
    header: JsonlV4Header = {
        "kind": "header",
        "version": 4,
        "id": _require_string(value.get("id"), path, 1, "id"),
        "createdAt": _require_timestamp(value.get("createdAt"), path, 1),
        "cwd": _require_string(value.get("cwd"), path, 1, "cwd"),
    }
    if parent_session_id is not None:
        header["parentSessionId"] = parent_session_id
    if legacy_parent_path is not None:
        header["legacyParentSessionPath"] = legacy_parent_path
    if metadata is not None:
        header["metadata"] = metadata
    return header


def encode_header(header: JsonlV4Header) -> str:
    return f"{json.dumps(header, ensure_ascii=False)}\n"


def metadata_from_header(
    header: JsonlV4Header, path: str, modified_at: int
) -> JsonlSessionMetadata:
    """由 header 派生会话元数据（对齐 TS metadataFromHeader）。"""
    metadata: JsonlSessionMetadata = {
        "id": header["id"],
        "createdAt": header["createdAt"],
        "cwd": header["cwd"],
        "path": path,
        "modifiedAt": modified_at,
        "sourceFormat": 4,
    }
    if header.get("parentSessionId") is not None:
        metadata["parentSessionId"] = header["parentSessionId"]
    if header.get("legacyParentSessionPath") is not None:
        metadata["legacyParentSessionPath"] = header["legacyParentSessionPath"]
    if header.get("metadata") is not None:
        metadata["metadata"] = header["metadata"]
    return metadata


def parse_mutation(line: str, path: str, line_number: int) -> SessionMutation:
    """解析一行 mutation（对齐 TS parseMutation）。"""
    value = _parse_object(line, path, line_number)
    seq = _require_sequence(value.get("seq"), path, line_number)
    kind = value.get("kind")

    if kind == "entry":
        entry_type = _require_string(value.get("type"), path, line_number, "entry type")
        if entry_type not in ENTRY_TYPES:
            raise invalid_file(path, line_number, f"has unknown entry type {entry_type}")
        entry: dict[str, Any] = {
            key: item for key, item in value.items() if key not in ("kind", "lane")
        }
        entry["id"] = _require_string(value.get("id"), path, line_number, "id")
        entry["type"] = entry_type
        entry["parentId"] = _require_nullable_id(
            value.get("parentId"), path, line_number, "parentId"
        )
        entry["seq"] = seq
        entry["timestamp"] = _require_timestamp(value.get("timestamp"), path, line_number)
        if entry_type == "custom":
            _require_string(value.get("customType"), path, line_number, "customType")
        entry = _entry_from_wire(entry)
        lane_raw = value.get("lane")
        if lane_raw is None:
            return {"kind": "entry", "entry": cast(Entry, entry)}
        lane = _require_string(lane_raw, path, line_number, "lane")
        return {"kind": "entry", "lane": lane, "entry": cast(Entry, entry)}

    if kind == "record":
        record_type = _require_string(value.get("type"), path, line_number, "record type")
        if record_type not in RECORD_TYPES:
            raise invalid_file(path, line_number, f"has unknown record type {record_type}")
        record: dict[str, Any] = {key: item for key, item in value.items() if key != "kind"}
        record["id"] = _require_string(value.get("id"), path, line_number, "id")
        record["lane"] = _require_string(value.get("lane"), path, line_number, "lane")
        record["type"] = record_type
        record["seq"] = seq
        record["timestamp"] = _require_timestamp(value.get("timestamp"), path, line_number)
        if record_type == "operation_started":
            intent = value.get("intent")
            if not isinstance(intent, dict):
                raise invalid_file(path, line_number, "has invalid intent")
            operation_kind = _require_string(
                intent.get("kind"), path, line_number, "operation kind"
            )
            if operation_kind not in OPERATION_KINDS:
                raise invalid_file(
                    path, line_number, f"has unknown operation kind {operation_kind}"
                )
        if record_type == "operation_finished":
            _require_string(value.get("runId"), path, line_number, "runId")
        if record_type == "usage" and isinstance(record.get("usage"), dict):
            record["usage"] = _usage_from_wire(record["usage"])
        return {"kind": "record", "record": cast(LaneRecord, record)}

    if kind == "lane":
        return {
            "kind": "lane",
            "seq": seq,
            "lane": _require_string(value.get("lane"), path, line_number, "lane"),
            "leafId": _require_nullable_id(value.get("leafId"), path, line_number, "leafId"),
        }

    if kind == "fact":
        fact = value.get("fact")
        if fact == "name":
            return {
                "kind": "fact",
                "seq": seq,
                "fact": "name",
                "name": _require_string(value.get("name"), path, line_number, "name"),
            }
        if fact == "label":
            label = value.get("label")
            if label is not None and not isinstance(label, str):
                raise invalid_file(path, line_number, "has invalid label")
            return {
                "kind": "fact",
                "seq": seq,
                "fact": "label",
                "targetId": _require_string(value.get("targetId"), path, line_number, "targetId"),
                "label": label,
            }
        raise invalid_file(path, line_number, "has unknown fact type")

    raise invalid_file(path, line_number, "has unknown mutation kind")


def encode_mutation(mutation: SessionMutation) -> str:
    """序列化一条 mutation（对齐 TS encodeMutation；message/usage 转 camelCase）。"""
    if mutation["kind"] == "entry":
        payload: dict[str, Any] = {"kind": "entry"}
        lane = mutation.get("lane")
        if lane is not None:
            payload["lane"] = lane
        payload.update(_entry_to_wire(mutation["entry"]))
    elif mutation["kind"] == "record":
        record = dict(mutation["record"])
        if record.get("type") == "usage" and isinstance(record.get("usage"), dict):
            record["usage"] = _usage_to_wire(record["usage"])
        payload = {"kind": "record", **record}
    else:
        payload = dict(mutation)
    return f"{json.dumps(payload, ensure_ascii=False)}\n"


__all__ = [
    "ENTRY_TYPES",
    "RECORD_TYPES",
    "OPERATION_KINDS",
    "invalid_file",
    "parse_header",
    "encode_header",
    "metadata_from_header",
    "parse_mutation",
    "encode_mutation",
]
