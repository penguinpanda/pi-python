"""DAG 会话树模型（Phase 3.1）。

对齐 TS `harness/session/session.ts`：

- 仅追加不变式：provider 上下文只在尾部增长
- `getBranch(fromId?)`：从根到指定节点（或最近的 compaction 起点）的路径
- `buildContext()`：构建 LLM 就绪上下文（自动跳过被压缩条目）
- `moveTo(entryId)`：移动 leaf（分支切换），可选附带 branch_summary
- 追加操作：message / thinking_level_change / model_change /
  active_tools_change / compaction / custom / custom_message / label /
  session_info
"""

from __future__ import annotations

import re
from typing import Any, Callable, cast

from pi_ai.types import ImageContent, TextContent, Usage

from .._types import AgentMessage

from .types import (
    ActiveToolsChangeEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionContext,
    SessionEntryCursorOptions,
    SessionError,
    SessionInfoEntry,
    SessionMetadata,
    SessionSnapshot,
    SessionStats,
    SessionStorage,
    SessionTreeEntry,
    ThinkingLevelChangeEntry,
)


ContextEntryTransform = Callable[[list[SessionTreeEntry]], list[SessionTreeEntry]]


class SessionContextBuildOptions:
    """buildContext 的附加选项。"""

    def __init__(
        self,
        entry_transforms: list[ContextEntryTransform] | None = None,
        entry_projectors: dict[
            str, Callable[[CustomEntry, int, list[SessionTreeEntry]], list[AgentMessage]]
        ]
        | None = None,
    ) -> None:
        self.entry_transforms = list(entry_transforms) if entry_transforms else []
        self.entry_projectors = dict(entry_projectors) if entry_projectors else {}


def _derive_session_context_state(
    path_entries: list[SessionTreeEntry],
) -> dict[str, Any]:
    """从路径条目推导 thinkingLevel / model / activeToolNames。"""
    thinking_level = "off"
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None

    for entry in path_entries:
        entry_type = entry["type"]
        if entry_type == "thinking_level_change":
            thinking_level = cast(ThinkingLevelChangeEntry, entry)["thinkingLevel"]
        elif entry_type == "model_change":
            change = cast(ModelChangeEntry, entry)
            model = {"provider": change["provider"], "modelId": change["modelId"]}
        elif entry_type == "message":
            message = cast(MessageEntry, entry)["message"]
            if message.get("role") == "assistant":
                model = {
                    "provider": str(message.get("provider", "")),
                    "modelId": str(message.get("model", "")),
                }
        elif entry_type == "active_tools_change":
            active_tool_names = list(cast(ActiveToolsChangeEntry, entry)["activeToolNames"])

    return {
        "thinkingLevel": thinking_level,
        "model": model,
        "activeToolNames": active_tool_names,
    }


def default_context_entry_transform(
    path_entries: list[SessionTreeEntry],
) -> list[SessionTreeEntry]:
    """默认压缩变换：保留最近一次 compaction 条目，跳过其摘要覆盖的历史。"""
    compaction: CompactionEntry | None = None
    for entry in path_entries:
        if entry["type"] == "compaction":
            compaction = cast(CompactionEntry, entry)
    if compaction is None:
        return list(path_entries)

    entries: list[SessionTreeEntry] = [cast(SessionTreeEntry, compaction)]
    compaction_index = next(
        (
            index
            for index, entry in enumerate(path_entries)
            if entry["type"] == "compaction" and entry["id"] == compaction["id"]
        ),
        -1,
    )
    if compaction.get("retainedTail"):
        entries.extend(path_entries[compaction_index + 1 :])
        return entries
    if compaction.get("firstKeptEntryId"):
        found_first_kept = False
        for entry in path_entries[:compaction_index]:
            if entry["id"] == compaction["firstKeptEntryId"]:
                found_first_kept = True
            if found_first_kept:
                entries.append(entry)
    entries.extend(path_entries[compaction_index + 1 :])
    return entries


def build_context_entries(
    path_entries: list[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> list[SessionTreeEntry]:
    entries = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms if options else []:
        entries = list(transform(entries))
    return entries


def _create_compaction_summary_message(
    summary: str,
    tokens_before: int,
    timestamp: str,
) -> AgentMessage:
    return cast(
        AgentMessage,
        {
            "role": "compactionSummary",
            "summary": summary,
            "tokensBefore": tokens_before,
            "timestamp": timestamp,
        },
    )


def _create_branch_summary_message(
    summary: str,
    from_id: str,
    timestamp: str,
) -> AgentMessage:
    return cast(
        AgentMessage,
        {
            "role": "branchSummary",
            "summary": summary,
            "fromId": from_id,
            "timestamp": timestamp,
        },
    )


def _create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: Any,
    timestamp: str,
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
    entry: SessionTreeEntry,
    index: int,
    entries: list[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> list[AgentMessage]:
    """把一条条目投影为 LLM 上下文消息。"""
    entry_type = entry["type"]
    if entry_type == "message":
        message = cast(MessageEntry, entry)["message"]
        # 对齐 TS v0.84：deferred 响应不进入 LLM 上下文。
        if message.get("role") == "assistant" and message.get("stopReason") == "deferred":
            return []
        return [message]
    if entry_type == "custom_message":
        custom = cast(CustomMessageEntry, entry)
        return [
            _create_custom_message(
                custom["customType"],
                custom["content"],
                custom["display"],
                custom.get("details"),
                custom["timestamp"],
            )
        ]
    if entry_type == "compaction":
        compaction = cast(CompactionEntry, entry)
        return [
            _create_compaction_summary_message(
                compaction["summary"],
                compaction.get("tokensBefore", 0),
                compaction["timestamp"],
            ),
            *(compaction.get("retainedTail") or []),
        ]
    if entry_type == "branch_summary":
        branch = cast(BranchSummaryEntry, entry)
        if branch.get("summary"):
            return [
                _create_branch_summary_message(
                    branch["summary"],
                    branch["fromId"],
                    branch["timestamp"],
                )
            ]
        return []
    if entry_type == "custom":
        custom_entry = cast(CustomEntry, entry)
        projector = (options.entry_projectors if options else {}).get(custom_entry["customType"])
        return list(projector(custom_entry, index, entries)) if projector else []
    return []


def build_session_context(
    path_entries: list[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> SessionContext:
    state = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages: list[AgentMessage] = []
    for index, entry in enumerate(context_entries):
        messages.extend(session_entry_to_context_messages(entry, index, context_entries, options))
    return cast(SessionContext, {**state, "messages": messages})


def _entries_by_id(
    entries: list[SessionTreeEntry],
) -> dict[str, SessionTreeEntry]:
    return {entry["id"]: entry for entry in entries}


def get_path_to_root_or_compaction(
    entries: list[SessionTreeEntry],
    leaf_id: str | None,
) -> list[SessionTreeEntry]:
    if leaf_id is None:
        return []
    by_id = _entries_by_id(entries)
    path: list[SessionTreeEntry] = []
    stop_at_entry_id: str | None = None
    current = by_id.get(leaf_id)
    if current is None:
        raise SessionError("not_found", f"Entry {leaf_id} not found")
    while current is not None:
        path.insert(0, current)
        if stop_at_entry_id is not None and current["id"] == stop_at_entry_id:
            break
        if current["type"] == "compaction":
            compaction = cast(CompactionEntry, current)
            if compaction.get("retainedTail"):
                break
            stop_at_entry_id = compaction.get("firstKeptEntryId") or None
        if not current.get("parentId"):
            break
        parent = by_id.get(current["parentId"])  # type: ignore[arg-type]
        if parent is None:
            raise SessionError(
                "invalid_session",
                f"Entry {current.get('parentId')} not found",
            )
        current = parent
    return path


def _get_label(entries: list[SessionTreeEntry], entry_id: str) -> str | None:
    label: str | None = None
    for entry in entries:
        if entry["type"] != "label":
            continue
        label_entry = cast(LabelEntry, entry)
        if label_entry["targetId"] != entry_id:
            continue
        trimmed = (label_entry.get("label") or "").strip()
        label = trimmed or None
    return label


def _get_session_name(entries: list[SessionTreeEntry]) -> str | None:
    for entry in reversed(entries):
        if entry["type"] == "session_info":
            name = cast(SessionInfoEntry, entry).get("name")
            trimmed = (name or "").strip()
            return trimmed or None
    return None


def _get_session_stats(entries: list[SessionTreeEntry]) -> SessionStats:
    message_count = 0
    cached_tokens = 0
    uncached_tokens = 0
    total_tokens = 0
    cost_total = 0.0
    for entry in entries:
        if entry["type"] == "message":
            message_count += 1
        usage: Any = None
        if entry["type"] == "message":
            message = cast(MessageEntry, entry)["message"]
            if message.get("role") == "assistant":
                usage = message.get("usage")
        elif entry["type"] in ("compaction", "branch_summary"):
            usage = entry.get("usage")
        if (
            not usage
            or not isinstance(usage.get("input"), (int, float))
            or not isinstance(usage.get("output"), (int, float))
            or not isinstance(usage.get("cacheRead"), (int, float))
            or not isinstance(usage.get("cacheWrite"), (int, float))
        ):
            continue
        cost = (usage.get("cost") or {}).get("total")
        if not isinstance(cost, (int, float)):
            continue
        cached_tokens += int(usage["cacheRead"])
        uncached_tokens += int(usage["input"]) + int(usage["cacheWrite"])
        total_tokens += (
            int(usage["input"])
            + int(usage["output"])
            + int(usage["cacheRead"])
            + int(usage["cacheWrite"])
        )
        cost_total += float(cost)
    return {
        "messageCount": message_count,
        "cachedTokens": cached_tokens,
        "uncachedTokens": uncached_tokens,
        "totalTokens": total_tokens,
        "costTotal": cost_total,
    }


def _timestamp_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class Session:
    """DAG 会话树模型。

    核心不变式：
    - 仅追加：provider 上下文只在尾部增长
    - 条目与编排记录严格分离
    - 预分配 ID 实现可恢复性
    """

    def __init__(
        self,
        storage: SessionStorage,
        context_build_options: SessionContextBuildOptions | None = None,
    ) -> None:
        self._storage = storage
        self._context_build_options = context_build_options or SessionContextBuildOptions()

    async def _load(self) -> SessionSnapshot:
        return {
            "metadata": await self._storage.get_metadata(),
            "leafId": await self._storage.get_leaf_id(),
            "entries": await self._storage.get_entries(),
        }

    async def get_metadata(self) -> SessionMetadata:
        return (await self._load())["metadata"]

    async def get_leaf_id(self) -> str | None:
        return (await self._load())["leafId"]

    async def get_entry(self, entry_id: str) -> SessionTreeEntry | None:
        return _entries_by_id((await self._load())["entries"]).get(entry_id)

    async def get_entries(
        self, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        return await self._storage.get_entries(options)

    async def get_branch(self, from_id: str | None = None) -> list[SessionTreeEntry]:
        """从根到指定节点（默认 leaf）的路径，到最近 compaction 起点为止。"""
        state = await self._load()
        return get_path_to_root_or_compaction(state["entries"], from_id or state["leafId"])

    def _merge_context_build_options(
        self, options: SessionContextBuildOptions | None
    ) -> SessionContextBuildOptions:
        return SessionContextBuildOptions(
            entry_transforms=[
                *(self._context_build_options.entry_transforms or []),
                *((options.entry_transforms or []) if options else []),
            ],
            entry_projectors={
                **(self._context_build_options.entry_projectors or {}),
                **((options.entry_projectors or {}) if options else {}),
            },
        )

    async def build_context_entries(
        self, options: SessionContextBuildOptions | None = None
    ) -> list[SessionTreeEntry]:
        return build_context_entries(
            await self.get_branch(),
            self._merge_context_build_options(options),
        )

    async def build_context(
        self, options: SessionContextBuildOptions | None = None
    ) -> SessionContext:
        return build_session_context(
            await self.get_branch(),
            self._merge_context_build_options(options),
        )

    async def get_label(self, entry_id: str) -> str | None:
        return _get_label((await self._load())["entries"], entry_id)

    async def get_session_stats(self) -> SessionStats:
        return _get_session_stats((await self._load())["entries"])

    async def get_session_name(self) -> str | None:
        return _get_session_name((await self._load())["entries"])

    async def _append_entry(self, entry: SessionTreeEntry) -> None:
        await self._storage.append_entry(entry)

    async def _create_entry_id(self) -> str:
        return await self._storage.create_entry_id()

    async def _append_typed(self, entry: SessionTreeEntry) -> str:
        await self._append_entry(entry)
        return entry["id"]

    async def append_message(self, message: AgentMessage) -> str:
        return await self._append_typed(
            {
                "type": "message",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "message": message,
            }
        )

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        return await self._append_typed(
            {
                "type": "thinking_level_change",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "thinkingLevel": thinking_level,
            }
        )

    async def append_model_change(self, provider: str, model_id: str) -> str:
        return await self._append_typed(
            {
                "type": "model_change",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "provider": provider,
                "modelId": model_id,
            }
        )

    async def append_active_tools_change(self, active_tool_names: list[str]) -> str:
        return await self._append_typed(
            {
                "type": "active_tools_change",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "activeToolNames": list(active_tool_names),
            }
        )

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Usage | None = None,
        retained_tail: list[AgentMessage] | None = None,
    ) -> str:
        entry: CompactionEntry = {
            "type": "compaction",
            "id": await self._create_entry_id(),
            "parentId": await self.get_leaf_id(),
            "timestamp": _timestamp_now(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        if details is not None:
            entry["details"] = details
        if from_hook is not None:
            entry["fromHook"] = from_hook
        if usage is not None:
            entry["usage"] = usage
        if retained_tail is not None:
            entry["retainedTail"] = list(retained_tail)
        return await self._append_typed(entry)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry: CustomEntry = {
            "type": "custom",
            "id": await self._create_entry_id(),
            "parentId": await self.get_leaf_id(),
            "timestamp": _timestamp_now(),
            "customType": custom_type,
        }
        if data is not None:
            entry["data"] = data
        return await self._append_typed(entry)

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent],
        display: bool,
        details: Any = None,
    ) -> str:
        entry: CustomMessageEntry = {
            "type": "custom_message",
            "id": await self._create_entry_id(),
            "parentId": await self.get_leaf_id(),
            "timestamp": _timestamp_now(),
            "customType": custom_type,
            "content": content,
            "display": display,
        }
        if details is not None:
            entry["details"] = details
        return await self._append_typed(entry)

    async def append_label(self, target_id: str, label: str | None) -> str:
        if await self.get_entry(target_id) is None:
            raise SessionError("not_found", f"Entry {target_id} not found")
        return await self._append_typed(
            {
                "type": "label",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "targetId": target_id,
                "label": label,
            }
        )

    async def append_session_name(self, name: str) -> str:
        sanitized = re.sub(r"[\r\n]+", " ", name).strip()
        return await self._append_typed(
            {
                "type": "session_info",
                "id": await self._create_entry_id(),
                "parentId": await self.get_leaf_id(),
                "timestamp": _timestamp_now(),
                "name": sanitized,
            }
        )

    async def move_to(
        self,
        entry_id: str | None,
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        """移动 leaf 到指定条目；可选附带 branch_summary 条目。"""
        if entry_id is not None and await self.get_entry(entry_id) is None:
            raise SessionError("not_found", f"Entry {entry_id} not found")
        await self._storage.set_leaf_id(entry_id)
        if not summary:
            return None
        entry: BranchSummaryEntry = {
            "type": "branch_summary",
            "id": await self._create_entry_id(),
            "parentId": entry_id,
            "timestamp": _timestamp_now(),
            "fromId": entry_id or "root",
            "summary": summary["summary"],
        }
        if summary.get("details") is not None:
            entry["details"] = summary["details"]
        if summary.get("usage") is not None:
            entry["usage"] = summary["usage"]
        if summary.get("fromHook") is not None:
            entry["fromHook"] = summary["fromHook"]
        return await self._append_typed(entry)
