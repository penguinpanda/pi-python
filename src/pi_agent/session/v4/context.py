"""v4 上下文构建（对齐 TS `harness/session/context.ts`）。"""

from __future__ import annotations

from typing import Any, Callable, cast

from ..._types import AgentMessage
from .types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    SessionContext,
)


ContextEntryTransform = Callable[[list[Entry]], list[Entry]]
CustomEntryContextMessageProjector = Callable[
    [CustomEntry, int, list[Entry]], list[AgentMessage] | None
]


class SessionContextBuildOptions:
    """build_context 的附加选项（对齐 TS SessionContextBuildOptions）。"""

    def __init__(
        self,
        entry_transforms: list[ContextEntryTransform] | None = None,
        entry_projectors: dict[str, CustomEntryContextMessageProjector] | None = None,
    ) -> None:
        self.entry_transforms = list(entry_transforms) if entry_transforms else []
        self.entry_projectors = dict(entry_projectors) if entry_projectors else {}


def _derive_session_context_state(
    path_entries: list[Entry],
) -> dict[str, Any]:
    """从路径条目推导 thinkingLevel / model / activeToolNames。"""
    thinking_level = "off"
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None
    for entry in path_entries:
        entry_type = entry["type"]
        if entry_type == "thinking_level_change":
            thinking_level = cast(Any, entry)["thinkingLevel"]
        elif entry_type == "model_change":
            change = cast(Any, entry)
            model = {"provider": change["provider"], "modelId": change["modelId"]}
        elif entry_type == "message" and cast(Any, entry)["message"].get("role") == "assistant":
            message = cast(Any, entry)["message"]
            model = {
                "provider": str(message.get("provider", "")),
                "modelId": str(message.get("model", "")),
            }
        elif entry_type == "active_tools_change":
            active_tool_names = list(cast(Any, entry)["activeToolNames"])
    return {
        "thinkingLevel": thinking_level,
        "model": model,
        "activeToolNames": active_tool_names,
    }


def default_context_entry_transform(path_entries: list[Entry]) -> list[Entry]:
    """保留最近一次 compaction 及其后的条目（v4 compaction 自带 retainedTail）。"""
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index]["type"] == "compaction":
            compaction_index = index
            break
    if compaction_index == -1:
        return list(path_entries)
    return [path_entries[compaction_index], *path_entries[compaction_index + 1 :]]


def _compaction_summary_message(summary: str, tokens_before: int, timestamp: int) -> AgentMessage:
    return cast(
        AgentMessage,
        {
            "role": "compactionSummary",
            "summary": summary,
            "tokensBefore": tokens_before,
            "timestamp": timestamp,
        },
    )


def _branch_summary_message(summary: str, from_id: str, timestamp: int) -> AgentMessage:
    return cast(
        AgentMessage,
        {
            "role": "branchSummary",
            "summary": summary,
            "fromId": from_id,
            "timestamp": timestamp,
        },
    )


def _custom_message(
    custom_type: str,
    content: Any,
    display: bool,
    details: Any,
    timestamp: int,
) -> AgentMessage:
    return cast(
        AgentMessage,
        {
            "role": "custom",
            "customType": custom_type,
            "content": content,
            "display": display,
            "details": details,
            "timestamp": timestamp,
        },
    )


def session_entry_to_context_messages(
    entry: Entry,
    index: int,
    entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[AgentMessage]:
    """把一条 v4 条目投影为 LLM 消息（对齐 TS sessionEntryToContextMessages）。"""
    entry_type = entry["type"]
    if entry_type == "message":
        message = cast(Any, entry)["message"]
        if message.get("role") == "assistant" and (
            message.get("stop_reason") == "deferred" or message.get("stopReason") == "deferred"
        ):
            return []
        return [message]
    if entry_type == "compaction":
        compaction = cast(CompactionEntry, entry)
        return [
            _compaction_summary_message(
                compaction["summary"], compaction["tokensBefore"], compaction["timestamp"]
            ),
            *compaction["retainedTail"],
        ]
    if entry_type == "branch_summary":
        branch = cast(BranchSummaryEntry, entry)
        if branch.get("summary"):
            return [
                _branch_summary_message(branch["summary"], branch["fromId"], branch["timestamp"])
            ]
        return []
    if entry_type == "custom":
        custom = cast(CustomEntry, entry)
        projector = (options.entry_projectors if options else {}).get(custom["customType"])
        projected = projector(custom, index, entries) if projector else None
        return list(projected) if projected else []
    return []


def build_context_entries(
    path_entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[Entry]:
    entries = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms if options else []:
        entries = list(transform(entries))
    return entries


def build_session_context(
    path_entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> SessionContext:
    """把路径条目构建为 LLM 就绪上下文（对齐 TS buildSessionContext）。"""
    state = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages: list[AgentMessage] = []
    for index, entry in enumerate(context_entries):
        messages.extend(session_entry_to_context_messages(entry, index, context_entries, options))
    return cast(SessionContext, {**state, "messages": messages})


__all__ = [
    "ContextEntryTransform",
    "CustomEntryContextMessageProjector",
    "SessionContextBuildOptions",
    "default_context_entry_transform",
    "build_context_entries",
    "session_entry_to_context_messages",
    "build_session_context",
]
