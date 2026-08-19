"""pi_coding_agent.compaction — 上下文自动压缩（移植 TS packages/coding-agent/src/core/compaction/compaction.ts + utils.ts）。

纯函数 + LLM 摘要压缩：

    estimate_tokens / estimate_context_tokens / should_compact   Token 估算与阈值
    find_cut_point / prepare_compaction                          切割点定位
    compact / generate_summary_with_usage                        LLM 摘要生成

设计：

- 纯函数操作"会话条目"（message 条目 + compaction 条目），与 SessionManager 解耦
- 摘要用独立 LLM 调用（cache_retention=none + 新 session_id），避免污染主上下文缓存
- 支持迭代压缩：previous_summary 复用 UPDATE 提示词合并
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, cast

from pi_agent import AgentMessage
from pi_agent.compaction import CompactionSettings, DEFAULT_COMPACTION_SETTINGS
from pi_agent.compaction_utils import (
    estimate_context_tokens,
    estimate_tokens,
    get_assistant_usage,
    should_compact,
)
from pi_ai import Context, Message, Model, Usage
from pi_ai.api._shared import empty_usage
from pi_ai.utils.retry import RetryPolicy, retry_assistant_call

from ._types import SessionEntry


def compaction_settings_from_config(settings: dict) -> CompactionSettings:
    """从 settings.json 的 `compaction` 节解析压缩配置（对齐 TS CompactionSettings）。

    支持的键：

        enabled           bool，默认 True
        reserveTokens     int，默认 16384（压缩阈值 = context_window - reserveTokens，
                          值越大越早压缩）
        keepRecentTokens  int，默认 20000（压缩时保留的最近 token 数）

    非法值回退默认。
    """
    raw = settings.get("compaction") if isinstance(settings, dict) else None
    if not isinstance(raw, dict):
        return DEFAULT_COMPACTION_SETTINGS
    base = DEFAULT_COMPACTION_SETTINGS

    enabled = raw.get("enabled", base.enabled)
    if not isinstance(enabled, bool):
        enabled = base.enabled

    reserve = raw.get("reserveTokens", base.reserve_tokens)
    if not isinstance(reserve, int) or reserve <= 0:
        reserve = base.reserve_tokens

    keep = raw.get("keepRecentTokens", base.keep_recent_tokens)
    if not isinstance(keep, int) or keep <= 0:
        keep = base.keep_recent_tokens

    return CompactionSettings(
        enabled=enabled,
        reserve_tokens=reserve,
        keep_recent_tokens=keep,
    )


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800
TOOL_RESULT_MAX_CHARS = 2000

# ---------------------------------------------------------------------------
# 压缩设置 / Token 估算：复用 pi_agent 的单一实现（CompactionSettings、
# estimate_tokens / estimate_context_tokens / should_compact / get_assistant_usage
# 从 pi_agent.compaction / pi_agent.compaction_utils 导入并原样再导出）。
# ---------------------------------------------------------------------------


def _get_last_assistant_usage_info(
    messages: list[AgentMessage],
) -> tuple[Usage, int] | None:
    """从后往前找最后一条有效 assistant usage（消息版，供 get_last_assistant_usage）。"""
    for i in range(len(messages) - 1, -1, -1):
        usage = get_assistant_usage(messages[i])
        if usage is not None:
            return usage, i
    return None


def get_last_assistant_usage(messages: list[AgentMessage]) -> Usage | None:
    """返回消息列表中最后一条有效 assistant usage（pi_agent 的版本按会话条目查找）。"""
    info = _get_last_assistant_usage_info(messages)
    return info[0] if info is not None else None


# ---------------------------------------------------------------------------
# 切割点检测（会话条目级）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CutPointResult:
    """切割点结果。"""

    # 第一条保留条目的下标。
    first_kept_entry_index: int
    # 被切分轮次的起始 user 消息下标；未切分时为 -1。
    turn_start_index: int
    # 是否在轮次中间切割。
    is_split_turn: bool


def is_cut_point_message(message: AgentMessage) -> bool:
    """该消息是否可作为切割点（绝不切在 toolResult 上）。"""
    role = message.get("role")
    if role in ("user", "assistant", "compactionSummary", "branchSummary", "custom"):
        return True
    if role == "toolResult":
        return False
    # agent 扩展角色（planner/observation/...）视为上下文可见 → 可切割。
    return role != "system"


def is_turn_start_message(message: AgentMessage) -> bool:
    """该消息是否为轮次起点（user-like）。"""
    return message.get("role") in ("user", "custom", "compactionSummary", "branchSummary")


def is_cut_point_entry(entry: dict[str, Any]) -> bool:
    """条目是否可作为切割点（仅 message 条目；compaction/配置类条目不可）。"""
    entry_type = entry.get("type")
    if entry_type == "compaction":
        return False
    if entry_type != "message":
        # thinking_level_change / model_change / active_tools_change / custom 等
        # 不承载 LLM 消息，不能作为切割点。
        return False
    return is_cut_point_message(entry["message"])


def is_turn_start_entry(entry: dict[str, Any]) -> bool:
    """条目是否为轮次起点（仅 message 条目）。"""
    if entry.get("type") != "message":
        return False
    return is_turn_start_message(entry["message"])


def _estimate_entry_tokens(entry: dict[str, Any]) -> int:
    """估算单个条目的 token 数（compaction 条目 = summary 文本）。"""
    entry_type = entry.get("type")
    if entry_type == "compaction":
        return math.ceil(len(entry.get("summary", "")) / CHARS_PER_TOKEN)
    if entry_type != "message":
        return 0
    return estimate_tokens(entry["message"])


def find_turn_start_index(entries: list[dict[str, Any]], entry_index: int, start_index: int) -> int:
    """向前（旧）查找包含 entry_index 所在轮次的 user-like 起点；找不到返回 -1。"""
    for i in range(entry_index, start_index - 1, -1):
        if is_turn_start_entry(entries[i]):
            return i
    return -1


def find_cut_point(
    entries: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """在条目中找切割点，使保留约 keep_recent_tokens 的近期上下文。

    从最新往旧遍历，累积估算 token，超过预算后取最近的合法切割点。
    可在 user 或 assistant 消息处切割（绝不切在 toolResult）；
    切在 assistant 消息中间时视为 split turn，其后跟随的 toolResult 会被保留。
    """
    cut_points = [i for i in range(start_index, end_index) if is_cut_point_entry(entries[i])]
    if not cut_points:
        return CutPointResult(start_index, -1, False)

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry_tokens = _estimate_entry_tokens(entries[i])
        if entry_tokens == 0:
            continue
        accumulated_tokens += entry_tokens
        if accumulated_tokens >= keep_recent_tokens:
            for c in cut_points:
                if c >= i:
                    cut_index = c
                    break
            break

    # 向后扫描，纳入不影响上下文的相邻条目。
    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.get("type") == "compaction":
            break
        if prev_entry.get("type") == "message":
            break  # 所有 message 条目都有上下文消息 → 停止回扫
        cut_index -= 1

    cut_entry = entries[cut_index]
    starts_turn = is_turn_start_entry(cut_entry)
    turn_start_index = -1 if starts_turn else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not starts_turn and turn_start_index != -1,
    )


# ---------------------------------------------------------------------------
# 文件操作追踪（摘要附加文件清单）
# ---------------------------------------------------------------------------


def create_file_ops() -> dict[str, set[str]]:
    """创建文件操作追踪集合。"""
    return {"read": set(), "written": set(), "edited": set()}


def extract_file_ops_from_message(message: AgentMessage, file_ops: dict[str, set[str]]) -> None:
    """从 assistant 消息的 toolCall 块提取文件操作。"""
    if message.get("role") != "assistant":
        return
    for block in cast(dict[str, Any], message).get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "toolCall":
            continue
        args = block.get("arguments")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
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
    """计算最终文件清单：readFiles（仅读）与 modifiedFiles（写/编辑）。"""
    modified = set(file_ops["edited"]) | set(file_ops["written"])
    read_only = sorted(f for f in file_ops["read"] if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """将文件清单格式化为 XML 标签追加到摘要。"""
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 对话序列化（供摘要）
# ---------------------------------------------------------------------------


def truncate_for_summary(text: str, max_chars: int) -> str:
    """截断文本到最大字符数（保留开头 + 截断标记）。"""
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def _content_text(content: Any, default: str = "") -> str:
    """提取内容块中的文本拼接（对齐 TS contentText）。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return default
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts) or default


def serialize_conversation(messages: list[AgentMessage]) -> str:
    """把 LLM 消息序列化为纯文本（防止模型把对话当作待继续内容）。

    工具结果截断到 TOOL_RESULT_MAX_CHARS，保证摘要请求在合理 token 预算内。
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            content = _content_text(msg.get("content"), "")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in cast(dict[str, Any], msg).get("content") or []:
                if not isinstance(block, dict):
                    continue
                block_dict = cast(dict[str, Any], block)
                btype = block_dict.get("type")
                if btype == "thinking":
                    thinking_parts.append(block_dict.get("thinking", ""))
                elif btype == "toolCall":
                    args = block_dict.get("arguments") or {}
                    args_str = ", ".join(
                        # separators 对齐 TS JSON.stringify 的无空格格式（{"x":1}）。
                        f"{k}={json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
                        for k, v in args.items()
                    )
                    tool_calls.append(f"{block_dict.get('name', '')}({args_str})")

            if thinking_parts:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinking_parts)}")
            if any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in (cast(dict[str, Any], msg).get("content") or [])
            ):
                parts.append(f"[Assistant]: {_content_text(msg.get('content'), '')}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif role == "toolResult":
            content = _content_text(msg.get("content"), "")
            if content:
                parts.append(
                    f"[Tool result]: {truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}"
                )
        elif role == "bashExecution":
            command = str(msg.get("command", ""))
            output = str(msg.get("output", ""))
            status = ""
            if msg.get("cancelled"):
                status = " (cancelled)"
            elif msg.get("exitCode") not in (None, 0):
                status = f" (exit {msg.get('exitCode')})"
            if output:
                parts.append(
                    f"[Bash]: {command}{status}\n{truncate_for_summary(output, TOOL_RESULT_MAX_CHARS)}"
                )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 摘要提示词（逐字对齐 TS）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompactionPreparation:
    """prepare_compaction 的产物（供 compact 使用）。"""

    first_kept_entry_id: str
    messages_to_summarize: list[AgentMessage]
    turn_prefix_messages: list[AgentMessage]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    settings: CompactionSettings


@dataclass(slots=True)
class CompactionResult:
    """compact 的结果（SessionManager 负责补 uuid/parentId 保存）。"""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    usage: Usage
    details: dict[str, Any] | None = None


def prepare_compaction(
    entries: list[SessionEntry],
    context_messages: list[AgentMessage],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """准备压缩：定位边界与切割点，收集待摘要消息。

    - 末条已是 compaction 条目 → 无可压缩内容，返回 None
    - 定位上一次压缩（previous_summary / boundary_start）
    - 无待摘要消息 → 返回 None
    """
    if entries and entries[-1].get("type") == "compaction":
        return None

    prev_compaction_index = -1
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        previous_summary = cast(str | None, prev.get("summary"))
        first_kept_index = next(
            (i for i, e in enumerate(entries) if e.get("id") == prev.get("firstKeptEntryId")),
            -1,
        )
        boundary_start = first_kept_index if first_kept_index >= 0 else prev_compaction_index + 1

    boundary_end = len(entries)
    tokens_before = estimate_context_tokens(context_messages).tokens

    cut = find_cut_point(
        [cast(dict[str, Any], e) for e in entries],
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens,
    )
    if cut.first_kept_entry_index >= len(entries):
        return None
    first_kept_entry = entries[cut.first_kept_entry_index]
    if not first_kept_entry.get("id"):
        return None  # 会话需要迁移
    first_kept_entry_id = first_kept_entry["id"]

    history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index
    messages_to_summarize = [
        cast(AgentMessage, e.get("message"))
        for e in entries[boundary_start:history_end]
        if e.get("type") == "message"
    ]
    turn_prefix_messages: list[AgentMessage] = []
    if cut.is_split_turn:
        turn_prefix_messages = [
            cast(AgentMessage, e.get("message"))
            for e in entries[cut.turn_start_index : cut.first_kept_entry_index]
            if e.get("type") == "message"
        ]

    if not messages_to_summarize and not turn_prefix_messages:
        return None

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        settings=settings,
    )


def _combine_usage(first: Usage, second: Usage) -> Usage:
    """合并两次摘要调用的 usage。"""

    def _g(u: Usage, key: str) -> int:
        return int(cast(Any, u).get(key, 0) or 0)

    def _cost(u: Usage) -> dict[str, float]:
        c = u.get("cost") or {}
        return {
            "input": c.get("input", 0) or 0,
            "output": c.get("output", 0) or 0,
            "cache_read": c.get("cache_read", 0) or 0,
            "cache_write": c.get("cache_write", 0) or 0,
            "total": c.get("total", 0) or 0,
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
    return result


async def complete_summarization(
    model: Model,
    context: Context,
    options: dict[str, Any],
    stream_fn,
    retry: RetryPolicy | None = None,
    callbacks=None,
):
    """摘要 LLM 调用的统一入口（可选 retry 包裹）。"""
    from pi_ai.types import AssistantMessage

    # 摘要为独立请求：隔离路由，避免不可复用的 cache 写入。
    request_options = {**options, "cache_retention": "none", "session_id": uuid.uuid4().hex}

    async def _produce() -> AssistantMessage:
        stream = await stream_fn(model, context, request_options)
        return await stream.result()

    if retry is not None:
        return await retry_assistant_call(
            _produce,
            policy=retry,
            signal=request_options.get("signal"),
            callbacks=callbacks,
        )
    return await _produce()


def _create_summarization_options(
    model: Model,
    max_tokens: int | float,
    api_key: str | None,
    thinking_level: str | None,
    *,
    headers: dict[str, str | None] | None = None,
    env: dict[str, str] | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """构建摘要请求选项（可显式覆盖认证：api_key / headers / env / base_url）。"""
    options: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if api_key:
        options["api_key"] = api_key
    if headers:
        options["headers"] = dict(headers)
    if env:
        options["env"] = dict(env)
    if base_url:
        options["base_url"] = base_url
    if thinking_level and thinking_level != "off":
        options["reasoning"] = thinking_level
    return options


async def generate_summary_with_usage(
    messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    api_key: str | None = None,
    headers: dict[str, str | None] | None = None,
    env: dict[str, str] | None = None,
    base_url: str | None = None,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: str | None = None,
    stream_fn,
    retry: RetryPolicy | None = None,
    callbacks=None,
) -> tuple[str, Usage]:
    """生成或更新对话摘要，返回 (文本, usage)。"""
    from pi_ai import now_ms

    max_tokens = min(
        math.floor(0.8 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )

    # 有 previous_summary 用更新提示词，否则用初始提示词。
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    conversation_text = serialize_conversation(messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    if custom_instructions:
        prompt_text += f"<custom-instructions>\n{custom_instructions}\n</custom-instructions>\n\n"
    prompt_text += base_prompt

    summarization_messages: list[Message] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": now_ms(),
        }
    ]

    completion_options = _create_summarization_options(
        model,
        max_tokens,
        api_key,
        thinking_level,
        headers=headers,
        env=env,
        base_url=base_url,
    )
    response = await complete_summarization(
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        stream_fn,
        retry,
        callbacks,
    )

    if response.get("stop_reason") == "error":
        raise RuntimeError(
            f"Summarization failed: {response.get('error_message') or 'Unknown error'}"
        )

    text = _content_text(response.get("content"), "")
    usage = response.get("usage") or empty_usage()
    return text, usage


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    api_key: str | None = None,
    headers: dict[str, str | None] | None = None,
    env: dict[str, str] | None = None,
    base_url: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: str | None = None,
    stream_fn,
    retry: RetryPolicy | None = None,
    callbacks=None,
) -> tuple[str, Usage]:
    """为被切分轮次的前缀生成摘要（预算更小：0.5 * reserve）。"""
    from pi_ai import now_ms

    max_tokens = min(
        math.floor(0.5 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )
    conversation_text = serialize_conversation(messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if custom_instructions:
        prompt_text += f"<custom-instructions>\n{custom_instructions}\n</custom-instructions>\n\n"
    prompt_text += TURN_PREFIX_SUMMARIZATION_PROMPT
    summarization_messages: list[Message] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": now_ms(),
        }
    ]

    response = await complete_summarization(
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        _create_summarization_options(
            model,
            max_tokens,
            api_key,
            thinking_level,
            headers=headers,
            env=env,
            base_url=base_url,
        ),
        stream_fn,
        retry,
        callbacks,
    )

    if response.get("stop_reason") == "error":
        raise RuntimeError(
            f"Turn prefix summarization failed: {response.get('error_message') or 'Unknown error'}"
        )

    return _content_text(response.get("content"), ""), response.get("usage") or empty_usage()


async def compact(
    preparation: CompactionPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    headers: dict[str, str | None] | None = None,
    env: dict[str, str] | None = None,
    base_url: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: str | None = None,
    stream_fn,
    retry: RetryPolicy | None = None,
    callbacks=None,
) -> CompactionResult:
    """基于 prepare_compaction 的结果执行压缩，返回 CompactionResult。"""
    first_kept_entry_id = preparation.first_kept_entry_id
    messages_to_summarize = preparation.messages_to_summarize
    turn_prefix_messages = preparation.turn_prefix_messages
    is_split_turn = preparation.is_split_turn
    tokens_before = preparation.tokens_before
    previous_summary = preparation.previous_summary
    settings = preparation.settings

    file_ops = create_file_ops()
    for msg in messages_to_summarize:
        extract_file_ops_from_message(msg, file_ops)
    if is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    if is_split_turn and turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if messages_to_summarize:
            history_text, history_usage = await generate_summary_with_usage(
                messages_to_summarize,
                model,
                settings.reserve_tokens,
                api_key=api_key,
                headers=headers,
                env=env,
                base_url=base_url,
                previous_summary=previous_summary,
                custom_instructions=custom_instructions,
                thinking_level=thinking_level,
                stream_fn=stream_fn,
                retry=retry,
                callbacks=callbacks,
            )
        turn_prefix_text, turn_prefix_usage = await _generate_turn_prefix_summary(
            turn_prefix_messages,
            model,
            settings.reserve_tokens,
            api_key=api_key,
            headers=headers,
            env=env,
            base_url=base_url,
            custom_instructions=custom_instructions,
            thinking_level=thinking_level,
            stream_fn=stream_fn,
            retry=retry,
            callbacks=callbacks,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_text}"
        summary_usage = (
            _combine_usage(history_usage, turn_prefix_usage) if history_usage else turn_prefix_usage
        )
    else:
        summary, summary_usage = await generate_summary_with_usage(
            messages_to_summarize,
            model,
            settings.reserve_tokens,
            api_key=api_key,
            headers=headers,
            env=env,
            base_url=base_url,
            previous_summary=previous_summary,
            custom_instructions=custom_instructions,
            thinking_level=thinking_level,
            stream_fn=stream_fn,
            retry=retry,
            callbacks=callbacks,
        )

    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)

    if not first_kept_entry_id:
        raise RuntimeError("First kept entry has no UUID - session may need migration")

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=first_kept_entry_id,
        tokens_before=tokens_before,
        usage=summary_usage,
        details={"readFiles": read_files, "modifiedFiles": modified_files},
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "ESTIMATED_IMAGE_CHARS",
    "TOOL_RESULT_MAX_CHARS",
    "CompactionSettings",
    "DEFAULT_COMPACTION_SETTINGS",
    "CutPointResult",
    "CompactionPreparation",
    "CompactionResult",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "get_assistant_usage",
    "estimate_tokens",
    "estimate_context_tokens",
    "get_last_assistant_usage",
    "should_compact",
    "is_cut_point_message",
    "is_turn_start_message",
    "find_turn_start_index",
    "find_cut_point",
    "create_file_ops",
    "extract_file_ops_from_message",
    "compute_file_lists",
    "format_file_operations",
    "truncate_for_summary",
    "serialize_conversation",
    "prepare_compaction",
    "complete_summarization",
    "generate_summary_with_usage",
    "compact",
]
