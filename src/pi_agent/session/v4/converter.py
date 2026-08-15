"""v3 → v4 会话惰性迁移（对齐 `session-v4-migration-plan.md` M2）。

读取 v3 JSONL（`pi_agent/session/v4/v3_reader.py`），转换为 v4 mutation 序列并
原子写回同一路径；原文件保留为 `<path>.bak`，转换失败不触碰原文件。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .v3_reader import (
    BranchSummaryEntry as V3BranchSummaryEntry,
    CompactionEntry as V3CompactionEntry,
    CustomEntry as V3CustomEntry,
    CustomMessageEntry as V3CustomMessageEntry,
    JsonlSessionStorage as V3JsonlSessionStorage,
    LabelEntry as V3LabelEntry,
    MessageEntry as V3MessageEntry,
    SessionError as V3SessionError,
    SessionInfoEntry as V3SessionInfoEntry,
    SessionTreeEntry as V3SessionTreeEntry,
)
from .codec import encode_header, encode_mutation, invalid_file
from .jsonl_types import JsonlSessionMetadata, JsonlV4Header
from .state import SessionMutation
from .storage import JsonlSessionStorage
from ..._types import AgentMessage
from .types import Entry, SessionError


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _iso_to_ms(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return 0
    return 0


def _parent_session_id_from_path(path: str) -> str | None:
    """尝试从 v3 parentSession 路径解析出会话 id；失败返回 None。"""
    match = re.search(r"_([A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])\.jsonl$", path)
    if match is None:
        return None
    candidate = match.group(1)
    return candidate if SESSION_ID_PATTERN.fullmatch(candidate) else None


def v3_header_metadata(first_line: str, path: str, modified_at: int) -> JsonlSessionMetadata:
    """解析 v3 首行，产出 v4 元数据（sourceFormat=3，供 list/open 使用）。"""
    try:
        value = json.loads(first_line)
    except json.JSONDecodeError as error:
        raise invalid_file(path, 1, "is not valid JSON", error) from error
    if not isinstance(value, dict) or value.get("type") != "session":
        raise invalid_file(path, 1, "is not a session header")
    if value.get("version") != 3:
        raise invalid_file(path, 1, "has unsupported session version")
    metadata: JsonlSessionMetadata = {
        "id": str(value.get("id") or ""),
        "createdAt": _iso_to_ms(value.get("timestamp")),
        "cwd": str(value.get("cwd") or ""),
        "path": path,
        "modifiedAt": modified_at,
        "sourceFormat": 3,
    }
    parent_session = value.get("parentSession")
    if isinstance(parent_session, str):
        parent_id = _parent_session_id_from_path(parent_session)
        if parent_id is not None:
            metadata["parentSessionId"] = parent_id
        else:
            metadata["legacyParentSessionPath"] = parent_session
    if isinstance(value.get("metadata"), dict):
        metadata["metadata"] = value["metadata"]
    return metadata


def _entry_mutation(entry: dict[str, Any], seq: int) -> SessionMutation:
    """把 v3 条目转换为 v4 entry mutation（不含 lane/fact）。"""
    entry_type = entry["type"]
    timestamp = _iso_to_ms(entry.get("timestamp"))
    result: dict[str, Any] = {
        "id": entry["id"],
        "parentId": entry.get("parentId"),
        "seq": seq,
        "timestamp": timestamp,
    }
    if entry_type == "message":
        message = cast_message(entry)
        result["type"] = "message"
        result["message"] = message["message"]
    elif entry_type == "thinking_level_change":
        result["type"] = "thinking_level_change"
        result["thinkingLevel"] = entry["thinkingLevel"]
    elif entry_type == "model_change":
        result["type"] = "model_change"
        result["provider"] = entry["provider"]
        result["modelId"] = entry["modelId"]
    elif entry_type == "active_tools_change":
        result["type"] = "active_tools_change"
        result["activeToolNames"] = list(entry["activeToolNames"])
    elif entry_type == "custom":
        custom = cast(V3CustomEntry, entry)
        result["type"] = "custom"
        result["customType"] = custom["customType"]
        if custom.get("data") is not None:
            result["data"] = custom["data"]
    elif entry_type == "branch_summary":
        branch = cast(V3BranchSummaryEntry, entry)
        result["type"] = "branch_summary"
        result["fromId"] = branch["fromId"]
        result["summary"] = branch["summary"]
        if branch.get("details") is not None:
            result["details"] = branch["details"]
        if branch.get("usage") is not None:
            result["usage"] = branch["usage"]
    elif entry_type == "custom_message":
        custom_message = cast(V3CustomMessageEntry, entry)
        result["type"] = "message"
        result["message"] = cast(
            AgentMessage,
            {
                "role": "custom",
                "customType": custom_message["customType"],
                "content": custom_message["content"],
                "display": bool(custom_message.get("display", True)),
                "details": custom_message.get("details"),
                "timestamp": timestamp,
            },
        )
    else:
        raise SessionError(
            "invalid_entry", f"Unsupported v3 entry type for migration: {entry_type}"
        )
    return {"kind": "entry", "entry": cast(Entry, result)}


def cast_message(entry: dict[str, Any]) -> V3MessageEntry:
    return entry  # type: ignore[return-value]


def _compaction_mutation(
    entry: V3CompactionEntry,
    seq: int,
    entries_after: list[V3SessionTreeEntry],
) -> SessionMutation:
    retained_tail = entry.get("retainedTail")
    if retained_tail is None:
        # 旧 v3 未存 retainedTail：按 firstKeptEntryId 从后续条目推导。
        retained_tail = []
        first_kept = entry.get("firstKeptEntryId")
        collecting = first_kept is None
        for after in entries_after:
            if after["id"] == first_kept:
                collecting = True
            if collecting and after["type"] == "message":
                retained_tail.append(after["message"])
    else:
        retained_tail = list(retained_tail)
    result: dict[str, Any] = {
        "type": "compaction",
        "id": entry["id"],
        "summary": entry["summary"],
        "retainedTail": list(retained_tail),
        "tokensBefore": int(entry.get("tokensBefore", 0) or 0),
        "parentId": entry.get("parentId"),
        "seq": seq,
        "timestamp": _iso_to_ms(entry.get("timestamp")),
    }
    if entry.get("details") is not None:
        result["details"] = entry["details"]
    if entry.get("usage") is not None:
        result["usage"] = entry["usage"]
    return {"kind": "entry", "entry": cast(Entry, result)}


def _v3_to_v4_mutations(
    header: dict[str, Any],
    entries: list[V3SessionTreeEntry],
    leaf_id: str | None,
) -> list[SessionMutation]:
    """按 v3 文件顺序生成 v4 mutation 序列（entry → lane → name/label facts）。"""
    mutations: list[SessionMutation] = []
    seq = 1
    name: str | None = None
    labels: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        entry_type = entry["type"]
        if entry_type == "compaction":
            mutations.append(
                _compaction_mutation(cast(V3CompactionEntry, entry), seq, entries[index + 1 :])
            )
            seq += 1
        elif entry_type in ("label", "session_info", "leaf"):
            if entry_type == "label":
                label = cast(V3LabelEntry, entry)
                labels.append(
                    {
                        "targetId": label["targetId"],
                        "label": label.get("label"),
                    }
                )
            elif entry_type == "session_info":
                info = cast(V3SessionInfoEntry, entry)
                candidate = (info.get("name") or "").strip()
                if candidate:
                    name = candidate
        else:
            mutations.append(_entry_mutation(cast(dict[str, Any], entry), seq))
            seq += 1

    mutations.append({"kind": "lane", "seq": seq, "lane": "main", "leafId": leaf_id})
    seq += 1
    if name is not None:
        mutations.append({"kind": "fact", "seq": seq, "fact": "name", "name": name})
        seq += 1
    for label_fact in labels:
        mutations.append(
            {
                "kind": "fact",
                "seq": seq,
                "fact": "label",
                "targetId": label_fact["targetId"],
                "label": label_fact["label"],
            }
        )
        seq += 1
    return mutations


def _v4_header(v3_header: dict[str, Any], cwd: str, parent_session: str | None) -> JsonlV4Header:
    header: JsonlV4Header = {
        "kind": "header",
        "version": 4,
        "id": str(v3_header.get("id") or ""),
        "createdAt": _iso_to_ms(v3_header.get("timestamp")),
        "cwd": cwd,
    }
    if parent_session is not None:
        parent_id = _parent_session_id_from_path(parent_session)
        if parent_id is not None:
            header["parentSessionId"] = parent_id
        else:
            header["legacyParentSessionPath"] = parent_session
    if isinstance(v3_header.get("metadata"), dict):
        header["metadata"] = v3_header["metadata"]
    return header


async def convert_v3_file_to_v4(path: str) -> JsonlSessionStorage:
    """把 v3 文件惰性转换为 v4（原子替换 + 原文件 .bak 备份）。"""
    file_path = Path(path)
    try:
        v3_storage = await V3JsonlSessionStorage.open(str(file_path))
    except V3SessionError as error:
        raise SessionError(
            "invalid_entry",
            f"Failed to read v3 session {path}: {error}",
            error,
        ) from error
    v3_header = cast(dict[str, Any], await v3_storage.get_metadata())
    entries = await v3_storage.get_entries()
    leaf_id = await v3_storage.get_leaf_id()

    parent_session: str | None = v3_header.get("parentSessionPath")
    cwd = str(v3_header.get("cwd") or "")
    header = _v4_header(v3_header, cwd, parent_session)
    mutations = _v3_to_v4_mutations(v3_header, entries, leaf_id)

    # 先在临时文件写入并回读校验；成功后再备份原文件并原子替换。
    def _populate(temp: str) -> None:
        temp_path = Path(temp)
        temp_path.write_text(encode_header(header), encoding="utf-8")
        with temp_path.open("a", encoding="utf-8") as handle:
            for mutation in mutations:
                handle.write(encode_mutation(mutation))

    temp_path = f"{path}.v4tmp"
    try:
        _populate(temp_path)
        await JsonlSessionStorage.load(temp_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

    backup_path = f"{path}.bak"
    os.replace(path, backup_path)
    try:
        os.replace(temp_path, path)
    except Exception:
        os.replace(backup_path, path)
        raise
    return await JsonlSessionStorage.load(path)


__all__ = [
    "convert_v3_file_to_v4",
    "v3_header_metadata",
    "_v3_to_v4_mutations",
]
