"""上下文压缩（Phase 4.1）。

对齐 TS `harness/compaction/compaction.ts`：基于 Session 条目做
prepare → LLM 摘要 → CompactionResult 的完整流程。
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable, cast

from pi_ai.types import Message, Usage
from pi_ai.utils.retry import RetryPolicy, retry_assistant_call

from ._types import AgentMessage
from .compaction_utils import (
    compute_file_lists,
    create_file_ops,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops_from_message,
    format_file_operations,
    get_message_from_entry_for_compaction,
    serialize_conversation,
)
from .session.session import build_session_context
from .session.types import CompactionEntry, SessionTreeEntry


class CompactionError(Exception):
    """Compaction 错误（对齐 TS CompactionError）。"""

    def __init__(
        self,
        code: str,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


@dataclass(slots=True)
class CompactionSettings:
    """压缩阈值与保留设置。"""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000
    cache_first: bool = False  # cache-first：优先截断低价值工具输出，保持前缀稳定


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


@dataclass(slots=True)
class CutPointResult:
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


def _find_valid_cut_points(
    entries: list[SessionTreeEntry],
    start_index: int,
    end_index: int,
) -> list[int]:
    cut_points: list[int] = []
    for index in range(start_index, end_index):
        entry = entries[index]
        if entry["type"] in ("branch_summary", "custom_message"):
            cut_points.append(index)
            continue
        if entry["type"] != "message":
            continue
        role = entry["message"].get("role")
        if role in (
            "bashExecution",
            "custom",
            "branchSummary",
            "compactionSummary",
            "user",
            "assistant",
        ):
            cut_points.append(index)
    return cut_points


def find_turn_start_index(
    entries: list[SessionTreeEntry],
    entry_index: int,
    start_index: int,
) -> int:
    for index in range(entry_index, start_index - 1, -1):
        entry = entries[index]
        if entry["type"] in ("branch_summary", "custom_message"):
            return index
        if entry["type"] == "message":
            if entry["message"].get("role") in ("user", "bashExecution"):
                return index
    return -1


def find_cut_point(
    entries: list[SessionTreeEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(start_index, -1, False)
    accumulated_tokens = 0
    cut_index = cut_points[0]
    for index in range(end_index - 1, start_index - 1, -1):
        entry = entries[index]
        if entry["type"] != "message":
            continue
        accumulated_tokens += estimate_tokens(entry["message"])
        if accumulated_tokens >= keep_recent_tokens:
            for candidate in cut_points:
                if candidate >= index:
                    cut_index = candidate
                    break
            break
    while cut_index > start_index:
        previous = entries[cut_index - 1]
        if previous["type"] in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry["type"] == "message" and cut_entry["message"].get("role") == "user"
    turn_start_index = (
        -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)
    )
    return CutPointResult(
        cut_index,
        turn_start_index,
        is_split_turn=(not is_user_message and turn_start_index != -1),
    )


SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def complete_simple_with_retries(
    stream_fn: Callable,
    model: Any,
    context: Any,
    options: dict[str, Any],
    retry: RetryPolicy | None = None,
    callbacks: Any = None,
):
    """摘要 LLM 调用（独立 session_id + cache_retention=none，可选 retry）。"""
    from pi_ai.types import AssistantMessage

    request_options = {
        **options,
        "cache_retention": "none",
        "session_id": uuid.uuid4().hex,
    }

    async def _produce() -> AssistantMessage:
        stream = await stream_fn(model, context, request_options)
        return await stream.result()

    return await retry_assistant_call(
        _produce,
        policy=retry,
        signal=request_options.get("signal"),
        callbacks=callbacks,
    )


def _combine_usage(first: Usage, second: Usage) -> Usage:
    def _g(usage: Usage, key: str) -> int:
        return int(cast(Any, usage).get(key, 0) or 0)

    def _cost(usage: Usage) -> dict[str, float]:
        cost = usage.get("cost") or {}
        return {
            "input": cost.get("input", 0) or 0,
            "output": cost.get("output", 0) or 0,
            "cache_read": cost.get("cache_read", 0) or 0,
            "cache_write": cost.get("cache_write", 0) or 0,
            "total": cost.get("total", 0) or 0,
        }

    c1, c2 = _cost(first), _cost(second)
    result: Usage = {
        "input": _g(first, "input") + _g(second, "input"),
        "output": _g(first, "output") + _g(second, "output"),
        "cache_read": _g(first, "cache_read") + _g(second, "cache_read"),
        "cache_write": _g(first, "cache_write") + _g(second, "cache_write"),
        "total_tokens": _g(first, "total_tokens") + _g(second, "total_tokens"),
        "cost": {
            "input": c1["input"] + c2["input"],
            "output": c1["output"] + c2["output"],
            "cache_read": c1["cache_read"] + c2["cache_read"],
            "cache_write": c1["cache_write"] + c2["cache_write"],
            "total": c1["total"] + c2["total"],
        },
    }
    if first.get("cache_write_1h") is not None or second.get("cache_write_1h") is not None:
        result["cache_write_1h"] = _g(first, "cache_write_1h") + _g(second, "cache_write_1h")
    if first.get("reasoning") is not None or second.get("reasoning") is not None:
        result["reasoning"] = _g(first, "reasoning") + _g(second, "reasoning")
    for raw_key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if first.get(raw_key) is not None or second.get(raw_key) is not None:
            result[raw_key] = _g(first, raw_key) + _g(second, raw_key)
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def generate_summary_with_usage(
    current_messages: list[AgentMessage],
    stream_fn: Callable,
    model: Any,
    reserve_tokens: int,
    signal: Any = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: Any = None,
) -> tuple[bool, Any]:
    """生成/更新摘要，返回 (ok, {text, usage} | CompactionError)。"""
    max_tokens = min(
        math.floor(0.8 * reserve_tokens),
        model.max_tokens if getattr(model, "max_tokens", 0) > 0 else float("inf"),
    )
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"
    conversation_text = serialize_conversation(current_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    from pi_ai import Context, now_ms

    summarization_messages: list[Message] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": now_ms(),
        }
    ]
    completion_options: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if signal is not None:
        completion_options["signal"] = signal
    if getattr(model, "reasoning", False) and thinking_level and thinking_level != "off":
        completion_options["reasoning"] = thinking_level

    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    stop_reason = response.get("stop_reason")
    if stop_reason == "aborted":
        return False, CompactionError(
            "aborted", response.get("error_message") or "Summarization aborted"
        )
    if stop_reason == "error":
        return False, CompactionError(
            "summarization_failed",
            f"Summarization failed: {response.get('error_message') or 'Unknown error'}",
        )
    return True, {
        "text": _content_text(response.get("content")),
        "usage": response.get("usage"),
    }


@dataclass(slots=True)
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[AgentMessage]
    turn_prefix_messages: list[AgentMessage]
    retained_tail: list[AgentMessage]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    file_ops: dict[str, set[str]]
    settings: CompactionSettings


def prepare_compaction(
    path_entries: list[SessionTreeEntry],
    settings: CompactionSettings,
) -> tuple[bool, Any]:
    """准备压缩；返回 (ok, preparation | None | CompactionError)。"""
    if not path_entries or path_entries[-1]["type"] == "compaction":
        return True, None

    prev_compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index]["type"] == "compaction":
            prev_compaction_index = index
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = cast_compaction(path_entries[prev_compaction_index])
        previous_summary = prev.get("summary")
        first_kept_entry_id = prev.get("firstKeptEntryId")
        first_kept_index = (
            next(
                (i for i, e in enumerate(path_entries) if e["id"] == first_kept_entry_id),
                -1,
            )
            if first_kept_entry_id
            else -1
        )
        boundary_start = first_kept_index if first_kept_index >= 0 else prev_compaction_index + 1

    boundary_end = len(path_entries)
    tokens_before = estimate_context_tokens(build_session_context(path_entries)["messages"]).tokens

    cut_point = find_cut_point(
        path_entries, boundary_start, boundary_end, settings.keep_recent_tokens
    )
    first_kept_entry = path_entries[cut_point.first_kept_entry_index]
    first_kept_entry_id = first_kept_entry.get("id")
    if not first_kept_entry_id:
        return False, CompactionError(
            "invalid_session",
            "First kept entry has no UUID - session may need migration",
        )

    history_end = (
        cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index
    )
    messages_to_summarize = [
        message
        for index in range(boundary_start, history_end)
        if (message := get_message_from_entry_for_compaction(path_entries[index])) is not None
    ]
    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        turn_prefix_messages = [
            message
            for index in range(cut_point.turn_start_index, cut_point.first_kept_entry_index)
            if (message := get_message_from_entry_for_compaction(path_entries[index])) is not None
        ]
    retained_tail: list[AgentMessage] = [
        message
        for index in range(cut_point.first_kept_entry_index, boundary_end)
        if (message := get_message_from_entry_for_compaction(path_entries[index])) is not None
    ]
    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for message in turn_prefix_messages:
            extract_file_ops_from_message(message, file_ops)

    return True, CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        retained_tail=retained_tail,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


def cast_compaction(entry: SessionTreeEntry) -> CompactionEntry:
    return entry  # type: ignore[return-value]


def _extract_file_operations(
    messages: list[AgentMessage],
    entries: list[SessionTreeEntry],
    prev_compaction_index: int,
) -> dict[str, set[str]]:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev = cast_compaction(entries[prev_compaction_index])
        if not prev.get("fromHook") and isinstance(prev.get("details"), dict):
            details = prev["details"]
            if isinstance(details.get("readFiles"), list):
                file_ops["read"].update(details["readFiles"])
            if isinstance(details.get("modifiedFiles"), list):
                file_ops["edited"].update(details["modifiedFiles"])
    for message in messages:
        extract_file_ops_from_message(message, file_ops)
    return file_ops


@dataclass(slots=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    usage: Usage | None
    retained_tail: list[AgentMessage] | None = None
    details: dict[str, Any] | None = None


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    stream_fn: Callable,
    model: Any,
    reserve_tokens: int,
    signal: Any = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: Any = None,
) -> tuple[bool, Any]:
    max_tokens = min(
        math.floor(0.5 * reserve_tokens),
        model.max_tokens if getattr(model, "max_tokens", 0) > 0 else float("inf"),
    )
    conversation_text = serialize_conversation(messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    from pi_ai import Context, now_ms

    summarization_messages: list[Message] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": now_ms(),
        }
    ]
    completion_options: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if signal is not None:
        completion_options["signal"] = signal
    if getattr(model, "reasoning", False) and thinking_level and thinking_level != "off":
        completion_options["reasoning"] = thinking_level
    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    stop_reason = response.get("stop_reason")
    if stop_reason == "aborted":
        return False, CompactionError(
            "aborted", response.get("error_message") or "Turn prefix summarization aborted"
        )
    if stop_reason == "error":
        return False, CompactionError(
            "summarization_failed",
            f"Turn prefix summarization failed: {response.get('error_message') or 'Unknown error'}",
        )
    return True, {
        "text": _content_text(response.get("content")),
        "usage": response.get("usage"),
    }


async def compact(
    preparation: CompactionPreparation,
    stream_fn: Callable,
    model: Any,
    custom_instructions: str | None = None,
    signal: Any = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: Any = None,
) -> tuple[bool, Any]:
    """执行压缩；返回 (ok, CompactionResult | CompactionError)。"""
    if not preparation.first_kept_entry_id:
        return False, CompactionError(
            "invalid_session",
            "First kept entry has no UUID - session may need migration",
        )

    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            ok_flag, result = await generate_summary_with_usage(
                preparation.messages_to_summarize,
                stream_fn,
                model,
                preparation.settings.reserve_tokens,
                signal,
                custom_instructions,
                preparation.previous_summary,
                thinking_level,
                retry,
                callbacks,
            )
            if not ok_flag:
                return False, result
            history_text = result["text"]
            history_usage = result["usage"]
        ok_flag, turn_prefix_result = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            stream_fn,
            model,
            preparation.settings.reserve_tokens,
            signal,
            thinking_level,
            retry,
            callbacks,
        )
        if not ok_flag:
            return False, turn_prefix_result
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_result['text']}"
        summary_usage = (
            _combine_usage(history_usage, turn_prefix_result["usage"])
            if history_usage
            else turn_prefix_result["usage"]
        )
    else:
        ok_flag, summary_result = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            stream_fn,
            model,
            preparation.settings.reserve_tokens,
            signal,
            custom_instructions,
            preparation.previous_summary,
            thinking_level,
            retry,
            callbacks,
        )
        if not ok_flag:
            return False, summary_result
        summary = summary_result["text"]
        summary_usage = summary_result["usage"]

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    return True, CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=summary_usage,
        retained_tail=preparation.retained_tail,
        details={"readFiles": read_files, "modifiedFiles": modified_files},
    )
