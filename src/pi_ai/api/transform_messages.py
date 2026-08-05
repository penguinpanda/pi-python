"""
跨 Provider 消息转换管道（Transform Messages）

对应 TS：

    packages/ai/src/api/transform-messages.ts

作用：

    发送给任何 Provider 之前，
    对历史消息做统一的规范化处理，
    保证不同 Provider 之间的消息兼容：

        ① 图片降级
           非视觉模型：image → 占位文本

        ② Thinking 块处理
           redacted / thinking_signature / 跨模型转换

        ③ Tool Call ID 规范化
           跨 API 工具调用 ID 哈希映射

        ④ 孤立 Tool Call 修复
           toolCall 没有对应 toolResult 时，
           自动补一条 error ToolResult

        ⑤ null content 归一化
           未类型化调用方（自定义工具、手工构造的历史、
           旧会话文件）可能缺失 content，统一补为 []

    与 `_shared.py` 的分工：

        _shared.py            → SDK Message → OpenAI Message（格式转换）
        transform_messages.py → 任意 Provider 发送前的消息规范化

    两者顺序：

        transform_messages(context.messages, model)
                    │
                    ▼
        to_openai_messages / _to_responses_input(...)
                    │
                    ▼
        Provider API
"""

import re
from typing import Any, Callable, cast

from ..types import (
    AssistantMessage,
    CodeContent,
    ContentBlock,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

# ------------------------------------------------------
# 图片降级占位文本
# ------------------------------------------------------

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


# ==========================================================
# short_hash — 确定性 32 位哈希（移植 TS utils/hash.ts）
# ==========================================================


def _to_base36(n: int) -> str:
    """将非负整数转换为 base36 字符串（对齐 JS Number.toString(36)）。"""

    if n == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


def short_hash(s: str) -> str:
    """快速确定性哈希，用于缩短长字符串。

    移植 TS `utils/hash.ts` 的 shortHash：

    - 两个 32 位哈希（h1/h2）各自用 imul 迭代；
    - 每次乘法都要 `& 0xFFFFFFFF`（Python int 不会自动溢出）；
    - `>>>16` / `>>>13` 先掩码再右移；
    - `charCodeAt` 是 UTF-16 code unit，需按 2 字节迭代
      （`s.encode("utf-16-le")`），否则非 ASCII 输入结果与 TS 不一致；
    - 输出 `(h2>>>0).toString(36) + (h1>>>0).toString(36)`。
    """

    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57

    data = s.encode("utf-16-le")
    for i in range(0, len(data), 2):
        ch = data[i] | (data[i + 1] << 8)
        h1 = ((h1 ^ ch) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ ch) * 1597334677) & 0xFFFFFFFF

    h1 = ((((h1 ^ (h1 >> 16)) & 0xFFFFFFFF) * 2246822507) & 0xFFFFFFFF) ^ (
        (((h2 ^ (h2 >> 13)) & 0xFFFFFFFF) * 3266489909) & 0xFFFFFFFF
    )
    h2 = ((((h2 ^ (h2 >> 16)) & 0xFFFFFFFF) * 2246822507) & 0xFFFFFFFF) ^ (
        (((h1 ^ (h1 >> 13)) & 0xFFFFFFFF) * 3266489909) & 0xFFFFFFFF
    )

    return _to_base36(h2 & 0xFFFFFFFF) + _to_base36(h1 & 0xFFFFFFFF)


# ==========================================================
# 图片降级
# ==========================================================


def replace_images_with_placeholder(
    content: list[dict[str, Any]],
    placeholder: str,
) -> list[dict[str, Any]]:
    """将内容块中的图片替换为占位文本。

    连续图片只插入一次占位符（去重），
    与 TS replaceImagesWithPlaceholder 一致。
    """

    result: list[dict[str, Any]] = []
    previous_was_placeholder = False

    for block in content:
        if block["type"] == "image":
            if not previous_was_placeholder:
                result.append({"type": "text", "text": placeholder})
            previous_was_placeholder = True
            continue

        result.append(block)
        previous_was_placeholder = block["type"] == "text" and block.get("text") == placeholder

    return result


def downgrade_unsupported_images(
    messages: list[Message],
    model: Model,
) -> list[Message]:
    """非视觉模型：将 user / toolResult 中的图片替换为占位文本。

    视觉模型（model.input 含 "image"）原样返回。

    注意：model.input 可能为 None（测试构造/部分模型），
    与 `_shared.py` 的写法保持一致。
    """

    if model.input and "image" in model.input:
        return messages

    result: list[Message] = []
    for msg in messages:
        role = msg["role"]

        # User 消息：仅数组 content 需要降级。
        if role == "user" and isinstance(msg.get("content"), list):
            new_msg = dict(msg)
            new_msg["content"] = replace_images_with_placeholder(
                msg["content"],  # type: ignore[arg-type]
                NON_VISION_USER_IMAGE_PLACEHOLDER,
            )
            result.append(new_msg)  # type: ignore[arg-type]

        # Tool 结果：同样支持图片。
        elif role == "toolResult":
            new_msg = dict(msg)
            new_msg["content"] = replace_images_with_placeholder(
                msg["content"],  # type: ignore[arg-type]
                NON_VISION_TOOL_IMAGE_PLACEHOLDER,
            )
            result.append(new_msg)  # type: ignore[arg-type]

        else:
            result.append(msg)

    return result


# ==========================================================
# Tool Call ID 规范化
# ==========================================================


def normalize_tool_call_id(
    id_: str,
    model: Model,
    source: AssistantMessage,
) -> str:
    """跨 Provider 工具调用 ID 规范化（对齐 TS openai-completions）。

    处理 Responses API 生成的管道分隔 ID：

        {call_id}|{item_id}

    其中 item_id 可达 400+ 字符且含特殊字符（+ / =），
    Chat Completions 要求工具调用 ID 互不相同且受限长度。

    规则：

    - 含 "|"：分离 call_id / item_id，各自 sanitize 为 [a-zA-Z0-9_-]，
      拼接 `call_id_item_id`；超过 40 字符用 short_hash 截断回退。
    - 不含 "|"：仅 openai provider 截断到 40 字符，其余原样。
    """

    if "|" in id_:
        call_id, _, item_id = id_.partition("|")
        call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
        item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item_id)
        combined_id = f"{call_id}_{item_id}" if item_id else call_id
        if len(combined_id) <= 40:
            return combined_id

        hash_part = short_hash(id_)[:8]
        prefix = call_id[: max(1, 40 - len(hash_part) - 1)]
        return f"{prefix}_{hash_part}"

    if model.provider == "openai":
        return id_[:40] if len(id_) > 40 else id_
    return id_


# ==========================================================
# Responses 系 Tool Call ID 规范化
# ==========================================================

# 生成管道分隔 ID（call_id|fc_item_id）的 Responses 系 provider。
#
# OpenAI Responses API 的 item id 以 "fc_" 开头（可达 400+ 字符、含特殊字符），
# 只有这些 provider 生成的 ID 需要保留双段结构；其他 provider 一律退化为单段。
ALLOWED_RESPONSES_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})


def normalize_id_part(part: str) -> str:
    """规范化 ID 单段：sanitize + 64 字符截断 + 去尾部下划线。"""

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", part)
    normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
    return normalized.rstrip("_")


def build_foreign_responses_item_id(item_id: str) -> str:
    """为跨 provider 的 item id 构建稳定的 fc_ 短 id。"""

    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64] if len(normalized) > 64 else normalized


def normalize_responses_tool_call_id(
    id_: str,
    model: Model,
    source: AssistantMessage,
) -> str:
    """Responses 系工具调用 ID 规范化（对齐 TS openai-responses-shared）。

    - 目标 provider 不在允许集合：整体退化为单段。
    - 无 "|"：单段规范化。
    - 有 "|"：call_id 规范化 + item_id 规范化；
      跨模型（source 非当前 provider/api）时用 short_hash 重建 fc_ item id；
      item id 必须 fc_ 开头。
    """

    if model.provider not in ALLOWED_RESPONSES_TOOL_CALL_PROVIDERS:
        return normalize_id_part(id_)

    if "|" not in id_:
        return normalize_id_part(id_)

    call_id, _, item_id = id_.partition("|")
    normalized_call_id = normalize_id_part(call_id)
    is_foreign_tool_call = (
        source.get("provider") != model.provider or source.get("api") != model.api
    )
    if is_foreign_tool_call:
        normalized_item_id = build_foreign_responses_item_id(item_id)
    else:
        normalized_item_id = normalize_id_part(item_id)

    if not normalized_item_id.startswith("fc_"):
        normalized_item_id = normalize_id_part(f"fc_{normalized_item_id}")

    return f"{normalized_call_id}|{normalized_item_id}"


# ==========================================================
# 主转换函数
# ==========================================================


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id_fn: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """跨 Provider 消息规范化管道（两阶段）。

    Parameters
    ----------
    messages
        发送前的历史消息。
    model
        目标模型元数据。
    normalize_tool_call_id_fn
        可选的工具调用 ID 规范化函数。

        仅在不同模型（跨 Provider / API）时生效，
        用于把上游 Provider 的特殊 ID 转换为当前 Provider 兼容格式。
    """

    # --------------------------------------------------
    # 预处理：null/missing content → []
    #
    # 未类型化调用方（自定义工具、手工构造历史、旧会话文件）
    # 可能违反类型契约；统一归一化，下游可依赖 content 是列表。
    # --------------------------------------------------
    normalized_messages = [
        cast(Message, {**msg, "content": []}) if msg.get("content") is None else msg
        for msg in messages
    ]

    image_aware_messages = downgrade_unsupported_images(normalized_messages, model)

    # 原始工具调用 ID → 规范化 ID 的映射。
    tool_call_id_map: dict[str, str] = {}

    def _is_same_model(asst: AssistantMessage) -> bool:
        """消息是否来自当前目标模型（provider/api/model 全同）。"""

        return (
            asst.get("provider") == model.provider
            and asst.get("api") == model.api
            and asst.get("model") == model.id
        )

    # --------------------------------------------------
    # 第一遍：逐条消息规范化
    #
    #   - 不支持图片降级（已在上方完成）
    #   - thinking 块处理
    #   - text 块跨模型重建（剥掉 text_signature）
    #   - toolCall ID 规范化
    # --------------------------------------------------
    transformed: list[Message] = []

    for msg in image_aware_messages:
        role = msg["role"]

        # User 消息原样透传。
        if role == "user":
            transformed.append(cast(UserMessage, msg))
            continue

        # toolResult：按映射规范化 tool_call_id。
        if role == "toolResult":
            tool_result = cast(ToolResultMessage, msg)
            original_id = tool_result.get("tool_call_id")
            normalized_id = tool_call_id_map.get(original_id or "")
            if normalized_id and normalized_id != original_id:
                new_msg = dict(tool_result)
                new_msg["tool_call_id"] = normalized_id
                transformed.append(new_msg)  # type: ignore[arg-type]
            else:
                transformed.append(tool_result)
            continue

        # Assistant 消息：需要检查 content 块。
        if role == "assistant":
            asst = cast(AssistantMessage, msg)
            same_model = _is_same_model(asst)

            transformed_content: list[ContentBlock] = []
            for content_block in asst["content"]:
                block_type = content_block["type"]

                # Thinking 块：
                #
                # - redacted 是加密不透明内容，仅同模型有效，跨模型丢弃
                # - 同模型 + thinking_signature：保留（即使 thinking 为空，
                #   例如 OpenAI 加密推理）
                # - 空 thinking 丢弃
                # - 其余：同模型保留，跨模型转为纯文本
                if block_type == "thinking":
                    thinking_block = cast(ThinkingContent, content_block)
                    if thinking_block.get("redacted"):
                        if same_model:
                            transformed_content.append(thinking_block)
                        continue

                    if same_model and thinking_block.get("thinking_signature"):
                        transformed_content.append(thinking_block)
                        continue

                    thinking = thinking_block.get("thinking") or ""
                    if not isinstance(thinking, str) or not thinking.strip():
                        continue

                    if same_model:
                        transformed_content.append(thinking_block)
                    else:
                        transformed_content.append(
                            cast(TextContent, {"type": "text", "text": thinking})
                        )
                    continue

                # Text 块：
                #
                # 同模型原样保留；
                # 跨模型重建 {type:text, text}（剥掉 text_signature）。
                if block_type == "text":
                    text_block = cast(TextContent, content_block)
                    if same_model:
                        transformed_content.append(text_block)
                    else:
                        transformed_content.append(
                            cast(
                                TextContent,
                                {"type": "text", "text": text_block.get("text", "")},
                            )
                        )
                    continue

                # Tool Call 块：
                #
                # - 跨模型删除 thought_signature（Google 专用签名）
                # - 跨模型 + normalize_tool_call_id_fn：规范化 ID 并记入映射
                if block_type == "toolCall":
                    tool_call = cast(ToolCall, content_block)
                    normalized_tool_call: dict[str, Any] = cast(dict[str, Any], tool_call)

                    if not same_model and tool_call.get("thought_signature"):
                        normalized_tool_call = dict(tool_call)
                        normalized_tool_call.pop("thought_signature", None)

                    if not same_model and normalize_tool_call_id_fn is not None:
                        normalized_id = normalize_tool_call_id_fn(tool_call["id"], model, asst)
                        if normalized_id != tool_call["id"]:
                            tool_call_id_map[tool_call["id"]] = normalized_id
                            if normalized_tool_call is tool_call:
                                normalized_tool_call = dict(tool_call)
                            normalized_tool_call["id"] = normalized_id

                    transformed_content.append(cast(ToolCall, normalized_tool_call))
                    continue

                # 未知块类型（code 等）原样透传。
                transformed_content.append(cast(CodeContent, content_block))

            new_msg = dict(asst)
            new_msg["content"] = transformed_content
            transformed.append(cast(Message, new_msg))
            continue

        # 其余 role（system / AgentMessage 等）原样透传。
        transformed.append(cast(Message, msg))

    # --------------------------------------------------
    # 第二遍：孤立 Tool Call 合成空 Tool Result
    #
    # 保留 thinking signature 并满足 API 要求：
    # 没有对应 toolResult 的 toolCall 会触发部分 API 报错。
    # --------------------------------------------------
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def insert_synthetic_tool_results() -> None:
        """为当前待处理的孤立 tool call 补合成错误结果。"""

        nonlocal pending_tool_calls, existing_tool_result_ids

        if pending_tool_calls:
            for tc in pending_tool_calls:
                if tc["id"] not in existing_tool_result_ids:
                    result.append(
                        {
                            "role": "toolResult",
                            "tool_call_id": tc["id"],
                            "tool_name": tc["name"],
                            "content": [{"type": "text", "text": "No result provided"}],
                            "is_error": True,
                            "timestamp": now_ms(),
                        }
                    )  # type: ignore[arg-type]
            pending_tool_calls = []
            existing_tool_result_ids = set()

    for msg in transformed:
        role = msg["role"]

        if role == "assistant":
            # 上一个 assistant 遗留的孤立 tool call，先补合成结果。
            insert_synthetic_tool_results()

            # error / aborted 的 assistant 整条跳过。
            #
            # 这些是不完整轮次，不应重放：
            # - 可能只有部分内容（有推理没有正文、工具调用不完整）
            # - 重放可能触发 API 报错（如 OpenAI "reasoning without following item"）
            # - 模型应从最后一个有效状态重试
            asst_msg = cast(AssistantMessage, msg)
            if asst_msg.get("stop_reason") in ("error", "aborted"):
                continue

            # 记录本轮的 tool call，供后续匹配 toolResult。
            tool_calls = [b for b in asst_msg["content"] if b["type"] == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(asst_msg)

        elif role == "toolResult":
            tool_result_msg = cast(ToolResultMessage, msg)
            existing_tool_result_ids.add(tool_result_msg["tool_call_id"])
            result.append(tool_result_msg)

        elif role == "user":
            # User 消息打断工具流：补合成孤立 tool call 的结果。
            insert_synthetic_tool_results()
            result.append(msg)

        else:
            # 其余 role（system / AgentMessage 的 planner/observation/... 等）
            # 直接透传，不触发孤立 tool call flush。
            #
            # 注意：TS 没有 AgentMessage 类型，仅 user 会打断工具流；
            # 这里刻意与 TS 语义保持一致（observation 等角色不打断），
            # 避免后人误以为是遗漏的 bug。
            result.append(msg)

    # 对话以未解决的 tool call 结束时，补合成结果。
    insert_synthetic_tool_results()

    return result


__all__ = [
    "NON_VISION_USER_IMAGE_PLACEHOLDER",
    "NON_VISION_TOOL_IMAGE_PLACEHOLDER",
    "short_hash",
    "replace_images_with_placeholder",
    "downgrade_unsupported_images",
    "normalize_tool_call_id",
    "ALLOWED_RESPONSES_TOOL_CALL_PROVIDERS",
    "normalize_id_part",
    "build_foreign_responses_item_id",
    "normalize_responses_tool_call_id",
    "transform_messages",
]
