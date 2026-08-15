"""pi_ai.utils.estimate — Token 估算（移植 TS packages/ai/src/utils/estimate.ts）。

为 Context 溢出检测 / max_tokens 收敛提供估算基础：

    ContextUsageEstimate
    ├── tokens           估算总上下文 token 数
    ├── usage_tokens     最近一次有效 assistant usage 报告的 token 数
    ├── trailing_tokens  usage 之后（尾部）消息的估算 token 数
    └── last_usage_index 提供 usage 的消息下标；无则 None

核心思路：只要对话中存在"最新且有效"的 assistant usage（成功响应），
就以它为准 + 尾部估算，避免对整段历史重复估算。
一旦在 usage 之前插入更新的消息（如 compaction 摘要），旧 usage 即失效
（它描述的是被替换前的旧前缀）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
)

# 每个 token 约等于的字符数（粗略估算）。
CHARS_PER_TOKEN = 4

# 单张图片的估算字符数（token 估算用）。
ESTIMATED_IMAGE_CHARS = 4800


@dataclass(slots=True)
class ContextUsageEstimate:
    """上下文用量估算结果。"""

    # 估算总上下文 token 数。
    tokens: int
    # 最近一次适用 assistant usage 报告的 token 数。
    usage_tokens: int
    # usage 之后（尾部）的估算 token 数。
    trailing_tokens: int
    # 提供 usage 的消息下标；无则 None。
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    """从 Usage 计算上下文 token 数。

    total_tokens 优先；缺失或为 0 时按 input/output/cache_read/cache_write 求和。
    """
    total = _as_int(usage.get("total_tokens"))
    if total:
        return total
    return (
        _as_int(usage.get("input"))
        + _as_int(usage.get("output"))
        + _as_int(usage.get("cache_read"))
        + _as_int(usage.get("cache_write"))
    )


def _as_int(value: Any) -> int:
    """usage 字段归一化：None/非数值（外部构造消息）按 0 处理。"""
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _safe_json_stringify(value: object) -> str:
    """JSON 序列化；不可序列化时返回占位字符串（对齐 TS safeJsonStringify）。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[unserializable]"


def _estimate_text_and_image_content_chars(content: str | list[TextContent | ImageContent]) -> int:
    """统计文本 + 图片内容的字符数。"""
    if isinstance(content, str):
        return len(content)

    chars = 0
    for block in content:
        chars += len(block["text"]) if block["type"] == "text" else ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    """估算纯文本的 token 数（每 CHARS_PER_TOKEN 字符约 1 token，向上取整）。"""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: str | list[TextContent | ImageContent]) -> int:
    """估算文本 + 图片内容块的 token 数。"""
    return math.ceil(_estimate_text_and_image_content_chars(content) / CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    """估算单条消息的 token 数。

    - user / toolResult：内容块（文本按长度、图片按固定估算值）
    - assistant：text / thinking 按长度，toolCall 按 名称 + 参数 JSON
    - system / agent：字符串内容按长度（Python 防御分支，TS 无）
    """
    if message["role"] == "user":
        return estimate_text_and_image_content_tokens(message.get("content") or [])
    if message["role"] == "toolResult":
        return estimate_text_and_image_content_tokens(message.get("content") or [])

    content = message.get("content")
    if isinstance(content, str):
        # system / agent 消息内容为纯字符串。
        return math.ceil(len(content) / CHARS_PER_TOKEN)

    chars = 0
    for block in content or []:
        if block["type"] == "text":
            chars += len(block["text"])
        elif block["type"] == "thinking":
            chars += len(block["thinking"])
        else:
            # toolCall（及未知块）：工具名 + 参数 JSON。
            chars += len(str(block.get("name") or "")) + len(
                _safe_json_stringify(block.get("arguments"))
            )
    return math.ceil(chars / CHARS_PER_TOKEN)


def _get_last_assistant_usage_info(messages: list[Message]) -> tuple[Usage, int] | None:
    """查找最新且适用于当前前缀的 assistant usage。

    规则（对齐 TS getLastAssistantUsageInfo）：

    - 仅 assistant 消息
    - 时间戳须不早于此前见过的最新时间戳（usage 之后若插入了更新的
      前缀消息，如 compaction 摘要，则该 usage 失效）
    - stop_reason 非 aborted / error
    - calculate_context_tokens(usage) > 0
    """
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            assistant = cast(AssistantMessage, message)
            usage_applies_to_prefix = _timestamp_ms(assistant.get("timestamp")) >= (
                latest_prefix_timestamp
            )
            usage = assistant.get("usage")
            if (
                usage_applies_to_prefix
                and assistant.get("stop_reason") not in ("aborted", "error")
                and usage is not None
                and calculate_context_tokens(usage) > 0
            ):
                usage_info = (usage, i)
        latest_prefix_timestamp = max(
            latest_prefix_timestamp, _timestamp_ms(message.get("timestamp"))
        )

    return usage_info


def _timestamp_ms(value: Any) -> float:
    """消息时间戳归一化：缺失/非数值按 -inf 处理（避免 str 比较 TypeError）。"""
    try:
        return float(value) if value is not None else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def _estimate_messages(messages: list[Message]) -> ContextUsageEstimate:
    """估算消息列表（无 system prompt / tools 前缀）。"""
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = 0
        for i in range(index + 1, len(messages)):
            trailing_tokens += estimate_message_tokens(messages[i])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing_tokens,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing_tokens,
            last_usage_index=index,
        )

    tokens = 0
    for message in messages:
        tokens += estimate_message_tokens(message)
    return ContextUsageEstimate(
        tokens=tokens,
        usage_tokens=0,
        trailing_tokens=tokens,
        last_usage_index=None,
    )


def _tool_to_dict(tool: Tool) -> dict[str, object]:
    """Tool dataclass → JSON 可序列化 dict（仅定义字段，排除 callable）。"""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def estimate_tools_tokens(tools: list[Tool] | None) -> int:
    """估算工具定义列表的 token 数。"""
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify([_tool_to_dict(t) for t in tools]))


def estimate_context_tokens(context: Context | list[Message]) -> ContextUsageEstimate:
    """估算完整上下文（Context 或消息列表）的 token 数。

    Context 分支：

    - 存在有效 usage 时：usage 之后 toolResult 新增声明的工具
      （added_tool_names）额外计入尾部
    - 无 usage 时：system prompt + tools 作为前缀计入
    """
    if isinstance(context, list):
        return _estimate_messages(context)

    estimate = _estimate_messages(context.messages)
    if estimate.last_usage_index is not None:
        added_names: set[str] = set()
        for message in context.messages[estimate.last_usage_index + 1 :]:
            if message["role"] == "toolResult":
                result = cast(ToolResultMessage, message)
                for name in result.get("added_tool_names") or []:
                    added_names.add(name)
        added_tool_tokens = estimate_tools_tokens(
            [tool for tool in context.tools if tool.name in added_names]
        )
        return ContextUsageEstimate(
            tokens=estimate.tokens + added_tool_tokens,
            usage_tokens=estimate.usage_tokens,
            trailing_tokens=estimate.trailing_tokens + added_tool_tokens,
            last_usage_index=estimate.last_usage_index,
        )

    prefix_tokens = (
        estimate_text_tokens(context.system_prompt) if context.system_prompt else 0
    ) + estimate_tools_tokens(context.tools)
    return ContextUsageEstimate(
        tokens=estimate.tokens + prefix_tokens,
        usage_tokens=estimate.usage_tokens,
        trailing_tokens=estimate.trailing_tokens + prefix_tokens,
        last_usage_index=estimate.last_usage_index,
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "ESTIMATED_IMAGE_CHARS",
    "ContextUsageEstimate",
    "calculate_context_tokens",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_text_tokens",
    "estimate_tools_tokens",
]
