"""Compaction 公共工具（Phase 4.1 辅助）。

对齐 TS `harness/compaction/utils.ts`：文件操作追踪、对话序列化、
token 估算（AgentMessage 级）。
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from pi_ai.types import Usage
from pi_ai.utils.estimate import ContextUsageEstimate

from ._types import AgentMessage
from .session.types import SessionTreeEntry
from .session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    MessageEntry,
)

TOOL_RESULT_MAX_CHARS = 2000
ESTIMATED_IMAGE_CHARS = 4800


def create_file_ops() -> dict[str, set[str]]:
    return {"read": set(), "written": set(), "edited": set()}


def extract_file_ops_from_message(message: AgentMessage, file_ops: dict[str, set[str]]) -> None:
    if message.get("role") != "assistant":
        return
    for block in cast(dict[str, Any], message).get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            continue
        arguments = block.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if not isinstance(path, str):
            continue
        name = block.get("name")
        if name == "read":
            file_ops["read"].add(path)
        elif name == "write":
            file_ops["written"].add(path)
        elif name == "edit":
            file_ops["edited"].add(path)


def compute_file_lists(file_ops: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    modified = set(file_ops["edited"]) | set(file_ops["written"])
    read_only = sorted(f for f in file_ops["read"] if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "[unserializable]"


def _content_text(content: Any, default: str = "") -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return default
    return (
        "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        or default
    )


def truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: list[AgentMessage]) -> str:
    """把 LLM 消息序列化为纯文本（防止模型继续对话）。"""
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            content = _content_text(message.get("content"), "")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in cast(dict[str, Any], message).get("content") or []:
                if not isinstance(block, dict):
                    continue
                block_dict = cast(dict[str, Any], block)
                block_type = block_dict.get("type")
                if block_type == "thinking":
                    thinking_parts.append(str(block_dict.get("thinking", "")))
                elif block_type == "toolCall":
                    args = block_dict.get("arguments") or {}
                    args_str = ", ".join(
                        f"{key}={safe_json_stringify(value)}" for key, value in args.items()
                    )
                    tool_calls.append(f"{block_dict.get('name', '')}({args_str})")
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinking_parts)}")
            if any(
                isinstance(block, dict) and block.get("type") == "text"
                for block in (cast(dict[str, Any], message).get("content") or [])
            ):
                parts.append(f"[Assistant]: {_content_text(message.get('content'), '')}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif role == "toolResult":
            content = _content_text(message.get("content"), "")
            if content:
                parts.append(
                    f"[Tool result]: {truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}"
                )
        elif role == "bashExecution":
            command = str(message.get("command", ""))
            output = str(message.get("output", ""))
            status = ""
            if message.get("cancelled"):
                status = " (cancelled)"
            elif message.get("exitCode") not in (None, 0):
                status = f" (exit {message.get('exitCode')})"
            if output:
                parts.append(
                    f"[Bash]: {command}{status}\n{truncate_for_summary(output, TOOL_RESULT_MAX_CHARS)}"
                )
    return "\n\n".join(parts)


def get_message_from_entry(entry: SessionTreeEntry) -> AgentMessage | None:
    """把会话条目投影为 AgentMessage（对齐 TS getMessageFromEntry）。"""
    from .session.session import (
        _create_branch_summary_message,
        _create_compaction_summary_message,
        _create_custom_message,
    )

    entry_type = entry["type"]
    if entry_type == "message":
        return cast(MessageEntry, entry)["message"]
    if entry_type == "custom_message":
        custom_entry = cast(CustomMessageEntry, entry)
        return _create_custom_message(
            custom_entry["customType"],
            custom_entry["content"],
            custom_entry["display"],
            custom_entry.get("details"),
            custom_entry["timestamp"],
        )
    if entry_type == "branch_summary":
        branch_entry = cast(BranchSummaryEntry, entry)
        return _create_branch_summary_message(
            branch_entry["summary"], branch_entry["fromId"], branch_entry["timestamp"]
        )
    if entry_type == "compaction":
        compaction_entry = cast(CompactionEntry, entry)
        return _create_compaction_summary_message(
            compaction_entry["summary"],
            compaction_entry.get("tokensBefore", 0),
            compaction_entry["timestamp"],
        )
    return None


def get_message_from_entry_for_compaction(entry: SessionTreeEntry) -> AgentMessage | None:
    if entry["type"] == "compaction":
        return None
    return get_message_from_entry(entry)


def estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and block.get("text"):
            chars += len(block["text"])
        elif block_type == "image":
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """保守的字符/token 估算（chars/4）。"""
    role = message.get("role")
    if role == "user":
        return math.ceil(estimate_text_and_image_content_chars(message.get("content") or "") / 4)
    if role == "assistant":
        chars = 0
        for block in cast(dict[str, Any], message).get("content") or []:
            if not isinstance(block, dict):
                continue
            block_dict = cast(dict[str, Any], block)
            block_type = block_dict.get("type")
            if block_type == "text":
                chars += len(block_dict.get("text", ""))
            elif block_type == "thinking":
                chars += len(block_dict.get("thinking", ""))
            elif block_type == "toolCall":
                chars += len(block_dict.get("name", "")) + len(
                    safe_json_stringify(block_dict.get("arguments") or {})
                )
        return math.ceil(chars / 4)
    if role in ("custom", "toolResult"):
        return math.ceil(estimate_text_and_image_content_chars(message.get("content") or "") / 4)
    if role in ("branchSummary", "compactionSummary"):
        return math.ceil(len(str(message.get("summary", ""))) / 4)
    if role == "bashExecution":
        return math.ceil(
            (len(str(message.get("command", ""))) + len(str(message.get("output", "")))) / 4
        )
    # 其余角色（system / agent 扩展角色）：优先按 content，其次按 summary 字段。
    content = message.get("content")
    if isinstance(content, str):
        return math.ceil(len(content) / 4)
    if isinstance(content, list):
        return math.ceil(estimate_text_and_image_content_chars(content) / 4)
    summary = message.get("summary")
    if isinstance(summary, str):
        return math.ceil(len(summary) / 4)
    return 0


def calculate_context_tokens(usage: Usage) -> int:
    return int(
        usage.get("total_tokens")
        or (
            int(usage.get("input", 0))
            + int(usage.get("output", 0))
            + int(usage.get("cache_read", 0))
            + int(usage.get("cache_write", 0))
        )
    )


def get_assistant_usage(message: AgentMessage) -> Usage | None:
    if message.get("role") != "assistant":
        return None
    usage = message.get("usage")
    if (
        message.get("stop_reason") not in ("aborted", "error")
        and isinstance(usage, dict)
        and calculate_context_tokens(cast(Usage, usage)) > 0
    ):
        return cast(Usage, usage)
    return None


def get_last_assistant_usage(entries: list[SessionTreeEntry]) -> Usage | None:
    for entry in reversed(entries):
        if entry["type"] == "message":
            usage = get_assistant_usage(entry["message"])
            if usage:
                return usage
    return None


def get_last_assistant_usage_info(messages: list[AgentMessage]) -> tuple[Usage, int] | None:
    for index in range(len(messages) - 1, -1, -1):
        usage = get_assistant_usage(messages[index])
        if usage:
            return usage, index
    return None


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    info = get_last_assistant_usage_info(messages)
    if info is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(estimated, 0, estimated, None)
    usage, index = info
    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = sum(estimate_tokens(message) for message in messages[index + 1 :])
    return ContextUsageEstimate(
        usage_tokens + trailing_tokens,
        usage_tokens,
        trailing_tokens,
        index,
    )


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: "Any",
) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens
