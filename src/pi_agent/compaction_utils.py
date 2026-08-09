"""Compaction 公共工具（Phase 4.1 辅助）。

对齐 TS `harness/compaction/utils.ts`：文件操作追踪、对话序列化、
token 估算（AgentMessage 级）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
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
    for block in message.get("content") or []:
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


CACHE_FIRST_TRUNCATED_MARKER = "[output truncated]"

# DeepSeek 上下文硬盘缓存 TTL（与 pi_coding_agent.cache_stats 同源；
# 实测 idle 6 分钟后缓存仍命中，5 分钟过于保守）。
CACHE_TTL_MS = 10 * 60 * 1000

# 保护尾预算：最近这些 token 内的工具结果不剪（对齐 Reasonix recentKeep）。
CACHE_FIRST_PROTECT_RECENT_TOKENS = 16_000

# 只读工具输出信息集中在前部，保留长头短尾；副作用工具头尾均衡。
_CACHE_FIRST_READ_ONLY_TOOLS = frozenset({"read", "grep", "rg", "find", "ls", "cat", "view"})
_CACHE_FIRST_HEAD_TAIL = {"read_only": (8000, 2000), "side_effect": (4000, 4000)}


def _snip_strategy_for(tool_name: str) -> tuple[int, int]:
    if tool_name in _CACHE_FIRST_READ_ONLY_TOOLS:
        return _CACHE_FIRST_HEAD_TAIL["read_only"]
    return _CACHE_FIRST_HEAD_TAIL["side_effect"]


def _snip_text(
    text: str,
    tool_name: str,
    archive_dir: Path | None = None,
) -> str:
    """head+tail 截断：保留头部与尾部，中间用稳定标记省略。"""
    head_chars, tail_chars = _snip_strategy_for(tool_name)
    if len(text) > head_chars + tail_chars:
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars else ""
    else:
        head_chars = len(text) // 2
        tail_chars = max(0, len(text) - head_chars - 1)
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars else ""
    omitted = len(text) - len(head) - len(tail)
    if omitted <= 0:
        return text
    archive_note = ""
    if archive_dir is not None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{digest}.txt"
        if not archive_path.exists():
            archive_path.write_text(text, encoding="utf-8")
        archive_note = f" archived={archive_path.name}"
    return (
        f"{CACHE_FIRST_TRUNCATED_MARKER}{archive_note}\n\n"
        f"{head}\n\n[... {omitted} chars omitted ...]\n\n{tail}"
    )


def _truncate_message(
    message: AgentMessage,
    archive_dir: Path | None = None,
) -> AgentMessage | None:
    """返回 head+tail 截断后的消息副本；已截断或无可截断内容返回 None。"""
    role = message.get("role")
    if role == "toolResult":
        content = message.get("content")
        tool_name = str(message.get("tool_name", ""))
        if isinstance(content, str):
            if len(content) <= TOOL_RESULT_MAX_CHARS or content.startswith(
                CACHE_FIRST_TRUNCATED_MARKER
            ):
                return None
            new_message = dict(message)
            new_message["content"] = _snip_text(content, tool_name, archive_dir)
            return cast(AgentMessage, new_message)
        if isinstance(content, list):
            text = _content_text(content, "")
            if len(text) <= TOOL_RESULT_MAX_CHARS or text.startswith(CACHE_FIRST_TRUNCATED_MARKER):
                return None
            new_message = dict(message)
            new_message["content"] = [
                {"type": "text", "text": _snip_text(text, tool_name, archive_dir)}
            ]
            return cast(AgentMessage, new_message)
        return None
    if role == "bashExecution":
        output = str(message.get("output", ""))
        if len(output) <= TOOL_RESULT_MAX_CHARS or output.startswith(CACHE_FIRST_TRUNCATED_MARKER):
            return None
        new_message = dict(message)
        new_message["output"] = _snip_text(output, "bash", archive_dir)
        return cast(AgentMessage, new_message)
    return None


def _truncate_all_large_tool_outputs(
    messages: list[AgentMessage],
    archive_dir: Path | None = None,
) -> list[AgentMessage]:
    result: list[AgentMessage] = []
    for message in messages:
        truncated = _truncate_message(message, archive_dir)
        result.append(truncated if truncated is not None else message)
    return result


def apply_cache_first_truncation(
    messages: list[AgentMessage],
    *,
    context_window: int = 0,
    reserve_tokens: int = 0,
    protect_recent_tokens: int = CACHE_FIRST_PROTECT_RECENT_TOKENS,
    archive_dir: str | Path | None = None,
) -> list[AgentMessage]:
    """对超长工具输出做 head+tail 截断（幂等、预算驱动、保护尾优先）。

    提供 context_window / reserve_tokens 时按预算处理：未超阈值不动；
    超阈值时优先截断保护尾之外的最新大输出，降到阈值内即停。
    保护尾（默认最近 16K token）内的结果保持完整，避免模型刚看到的内容被抹掉。
    archive_dir 非空时截断前把原始内容归档到该目录。
    """
    resolved_archive = Path(archive_dir) if archive_dir is not None else None
    if context_window <= 0 or reserve_tokens < 0:
        return _truncate_all_large_tool_outputs(messages, resolved_archive)
    threshold = context_window - reserve_tokens

    def _estimate(messages: list[AgentMessage]) -> int:
        # 全量字符估算：剪枝只改视图，usage 反映的是上次已剪枝的请求，
        # 用它做预算会漏掉 state 里仍完整的旧工具输出，导致时剪时不剪。
        return sum(estimate_tokens(message) for message in messages)

    if threshold <= 0 or _estimate(messages) <= threshold:
        return messages
    result = list(messages)
    if protect_recent_tokens <= 0:
        prunable_indices = range(len(result) - 1, -1, -1)
    else:
        protect_start = 0
        accumulated = 0
        for index in range(len(result) - 1, -1, -1):
            accumulated += estimate_tokens(result[index])
            if accumulated >= protect_recent_tokens:
                protect_start = index
                break
        prunable_indices = range(protect_start - 1, -1, -1)
    for index in prunable_indices:
        truncated = _truncate_message(result[index], resolved_archive)
        if truncated is None:
            continue
        result[index] = truncated
        if _estimate(result) <= threshold:
            break
    return result


def estimate_cache_state(messages: list[AgentMessage], now_ms: int | None = None) -> str:
    """按最后一条有 usage 的 assistant 消息时间估算 warm/cold/unknown。"""
    import time

    now = int(time.time() * 1000) if now_ms is None else now_ms
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict) or not usage:
            continue
        timestamp = message.get("timestamp")
        if not isinstance(timestamp, int):
            return "unknown"
        return "warm" if now - timestamp <= CACHE_TTL_MS else "cold"
    return "unknown"


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
            for block in message.get("content") or []:
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
                for block in (message.get("content") or [])
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
        for block in message.get("content") or []:
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
        return math.ceil(len(message.get("summary", "")) / 4)
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
