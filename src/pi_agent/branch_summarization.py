"""分支摘要（Phase 4.2）。

对齐 TS `harness/compaction/branch-summarization.ts`：从旧 leaf 切换到新
target 前，收集差异条目并生成结构化摘要。
"""

from __future__ import annotations

import math
from typing import Any

from pi_ai._types import AgentMessage

from .compaction import SUMMARIZATION_SYSTEM_PROMPT, complete_simple_with_retries
from .compaction_utils import (
    compute_file_lists,
    create_file_ops,
    estimate_tokens,
    extract_file_ops_from_message,
    format_file_operations,
    get_message_from_entry,
    serialize_conversation,
)
from .session.session import _create_branch_summary_message, _create_compaction_summary_message, _create_custom_message
from .session.types import SessionError, SessionTreeEntry


class BranchSummaryError(Exception):
    """分支摘要错误（对齐 TS BranchSummaryError）。"""

    def __init__(
        self,
        code: str,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


async def collect_entries_for_branch_summary(
    session: Any,
    old_leaf_id: str | None,
    target_id: str,
) -> dict[str, Any]:
    """收集旧 leaf → 共同祖先的差异条目。"""
    if not old_leaf_id:
        return {"entries": [], "commonAncestorId": None}
    old_path = {entry["id"] for entry in await session.get_branch(old_leaf_id)}
    target_path = await session.get_branch(target_id)
    common_ancestor_id: str | None = None
    for index in range(len(target_path) - 1, -1, -1):
        if target_path[index]["id"] in old_path:
            common_ancestor_id = target_path[index]["id"]
            break
    entries: list[SessionTreeEntry] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if entry is None:
            raise SessionError("invalid_session", f"Entry {current} not found")
        entries.append(entry)
        current = entry.get("parentId")
    entries.reverse()
    return {"entries": entries, "commonAncestorId": common_ancestor_id}


def _get_message_from_entry(entry: SessionTreeEntry) -> AgentMessage | None:
    if entry["type"] == "message":
        if entry["message"].get("role") == "toolResult":
            return None
        return entry["message"]
    if entry["type"] == "custom_message":
        return _create_custom_message(
            entry["customType"],
            entry["content"],
            entry["display"],
            entry.get("details"),
            entry["timestamp"],
        )
    if entry["type"] == "branch_summary":
        return _create_branch_summary_message(entry["summary"], entry["fromId"], entry["timestamp"])
    if entry["type"] == "compaction":
        return _create_compaction_summary_message(
            entry["summary"], entry.get("tokensBefore", 0), entry["timestamp"]
        )
    return None


def prepare_branch_entries(
    entries: list[SessionTreeEntry],
    token_budget: int = 0,
) -> dict[str, Any]:
    """在 token 预算内准备分支摘要消息。"""
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0
    for entry in entries:
        if entry["type"] == "branch_summary" and not entry.get("fromHook") and isinstance(entry.get("details"), dict):
            details = entry["details"]
            if isinstance(details.get("readFiles"), list):
                for file_path in details["readFiles"]:
                    file_ops["read"].add(file_path)
            if isinstance(details.get("modifiedFiles"), list):
                for file_path in details["modifiedFiles"]:
                    file_ops["edited"].add(file_path)
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)
        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry["type"] in ("compaction", "branch_summary"):
                if total_tokens < token_budget * 0.9:
                    messages.insert(0, message)
                    total_tokens += tokens
            break
        messages.insert(0, message)
        total_tokens += tokens
    return {"messages": messages, "fileOps": file_ops, "totalTokens": total_tokens}


BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[SessionTreeEntry],
    *,
    stream_fn: Any,
    model: Any,
    signal: Any = None,
    custom_instructions: str | None = None,
    replace_instructions: bool = False,
    reserve_tokens: int = 16384,
    retry: Any = None,
    callbacks: Any = None,
) -> tuple[bool, Any]:
    """生成分支摘要；返回 (ok, {summary, usage, readFiles, modifiedFiles} | BranchSummaryError)。"""
    context_window = getattr(model, "context_window", 0) or 128000
    token_budget = context_window - reserve_tokens
    prepared = prepare_branch_entries(entries, token_budget)
    messages = prepared["messages"]
    file_ops = prepared["fileOps"]

    if not messages:
        return True, {"summary": "No content to summarize", "readFiles": [], "modifiedFiles": []}

    conversation_text = serialize_conversation(messages)
    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"

    from pi_ai import Context, now_ms

    summarization_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": now_ms(),
        }
    ]
    options: dict[str, Any] = {"max_tokens": 2048}
    if signal is not None:
        options["signal"] = signal
    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        options,
        retry,
        callbacks,
    )
    stop_reason = response.get("stop_reason")
    if stop_reason == "aborted":
        return False, BranchSummaryError("aborted", response.get("error_message") or "Branch summary aborted")
    if stop_reason == "error":
        return False, BranchSummaryError(
            "summarization_failed",
            f"Branch summary failed: {response.get('error_message') or 'Unknown error'}",
        )

    summary = _content_text(response.get("content"))
    summary = BRANCH_SUMMARY_PREAMBLE + summary
    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)
    return True, {
        "summary": summary or "No summary generated",
        "usage": response.get("usage"),
        "readFiles": read_files,
        "modifiedFiles": modified_files,
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
