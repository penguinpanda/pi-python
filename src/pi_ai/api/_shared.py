"""
API 实现共享工具(Shared Helpers)

SDK 内部类型
        │
        ▼
_shared.py(格式转换)
        │
        ▼
OpenAI API 类型

=========================================================
模块职责
=========================================================

本模块提供多个 Provider 共用的辅助函数。

主要包括：

    ① 消息格式转换

        SDK Message
                │
                ▼
        OpenAI Message

    ② Tool 定义转换

        SDK Tool
                │
                ▼
        OpenAI Tool

    ③ Usage 构造

    ④ 错误消息构造

    ⑤ ContentBlock 文本提取

    ⑥ Tool Call 增量拼接

这些函数不负责：

    - 网络请求
    - EventStream
    - Provider 调度

仅负责不同数据结构之间的转换。
"""

import json
from typing import Any, cast

from ..types import (
    AssistantMessage,
    ContentBlock,
    Message,
    Model,
    SystemMessage,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from ..utils.diagnostics import create_assistant_message_diagnostic
from .constrained_sampling import resolve_json_schema_strict_sampling


def to_openai_messages(
    messages: list[Message],
    model: Model,
) -> list[dict[str, Any]]:
    """
    将 SDK Message

    转换为 OpenAI Chat Completions Message。

    例如：

    SDK

    UserMessage

    ↓

    {
        "role":"user",
        "content":[
            TextContent,
            ImageContent
        ]
    }

    ↓

    OpenAI

    {
        "role":"user",
        "content":[
            ...
        ]
    }

    不同 Role

    会分别转换：

    system

    ↓

    system

    user

    ↓

    user

    assistant

    ↓

    assistant

    toolResult

    ↓

    tool
    """
    # OpenAI Message 列表。
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        # System Message
        #
        # 两种 SDK 格式一致，
        # 直接转换。
        if role == "system":
            sys_msg = cast(SystemMessage, msg)
            result.append({"role": "system", "content": sys_msg["content"]})

        # User Message
        #
        # 用户输入既可以：
        #
        # ① 字符串
        #
        # "Hello"
        #
        # ② 多模态
        #
        # [
        #     TextContent,
        #     ImageContent
        # ]
        #
        # 根据不同内容，
        # 转换为 OpenAI Message。
        elif role == "user":
            user_msg = cast(UserMessage, msg)
            content = user_msg["content"]
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            else:
                parts: list[dict[str, Any]] = []
                for block in content:
                    if block["type"] == "text":
                        parts.append({"type": "text", "text": block["text"]})

                    # 图片输入。
                    #
                    # 只有模型支持 image 输入时，
                    # 才转换图片内容。
                    elif block["type"] == "image" and model.input and "image" in model.input:
                        image_part: dict[str, Any] = {"type": "image_url"}
                        if block.get("url"):
                            image_part["image_url"] = {"url": block["url"]}
                        elif block.get("data"):
                            # Base64 图片需要转换成
                            #
                            # data URL。
                            image_part["image_url"] = {
                                "url": f"data:{block.get('mime_type', 'image/png')};base64,{block['data']}"
                            }

                        parts.append(image_part)

                if len(parts) == 1 and parts[0]["type"] == "text":
                    result.append({"role": "user", "content": parts[0]["text"]})
                else:
                    result.append({"role": "user", "content": parts})

        # Assistant Message。
        #
        # 转换：
        #
        # Text
        #
        # Tool Call
        #
        # Thinking
        #
        # Thinking 不发送给 OpenAI，
        # 因为属于 Provider 内部信息。
        elif role == "assistant":
            asst_msg = cast(AssistantMessage, msg)
            oai_msg: dict[str, Any] = {"role": "assistant", "content": None}
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for asst_block in asst_msg["content"]:
                if asst_block["type"] == "text":
                    text_parts.append(asst_block["text"])
                elif asst_block["type"] == "toolCall":
                    tool_calls.append(
                        {
                            "id": asst_block["id"],
                            "type": "function",
                            "function": {
                                "name": asst_block["name"],
                                "arguments": json.dumps(
                                    asst_block["arguments"]
                                    if asst_block["arguments"] is not None
                                    else {},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                elif asst_block["type"] == "thinking":
                    # Skip assistant thinking blocks — replayed as text
                    pass

            # OpenAI Tool Calling
            #
            # 使用 tool_calls 字段。
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            if text_parts:
                oai_msg["content"] = "\n".join(text_parts)

            result.append(oai_msg)

        # Tool 返回结果。
        #
        # SDK：
        #
        # toolResult
        #
        # OpenAI：
        #
        # tool
        elif role == "toolResult":
            tr_msg = cast(ToolResultMessage, msg)
            content_str = ""
            for block in tr_msg["content"]:
                if block["type"] == "text":
                    content_str += block["text"]

            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tr_msg["tool_call_id"],
                    "content": content_str,
                }
            )

    return result


def to_openai_tools(
    tools: list[Tool],
    *,
    supports_strict_mode: bool = True,
) -> list[dict[str, Any]]:
    """
    将 SDK Tool

    转换为 OpenAI Tool Schema。

    例如：

    Tool

    ↓

    {
        "type":"function",
        ...
    }
    """
    result: list[dict[str, Any]] = []
    for t in tools:
        strict = resolve_json_schema_strict_sampling(t, supports_strict_mode)
        function_schema: dict[str, Any] = {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
        if strict:
            function_schema["strict"] = True
        result.append(
            {
                "type": "function",
                "function": function_schema,
            }
        )
    return result


def empty_usage() -> Usage:
    """
    创建一个空 Usage。

    用于：

    - 请求失败

    - Provider 尚未返回 Usage

    保证 AssistantMessage

    始终拥有 Usage 字段。
    """

    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )


def build_error_message(
    model: Model,
    error: Exception,
) -> AssistantMessage:
    """
    构造错误 AssistantMessage。

    当 Provider 请求失败时，

    SDK 统一返回：

    AssistantMessage

    而不是直接返回 Exception。

    这样：

    正常消息

    错误消息

    拥有一致的数据结构。
    """

    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=empty_usage(),
        stop_reason="error",
        error_message=str(error),
        # 结构化诊断：provider 错误定位（对齐 TS 各 provider 的
        # createAssistantMessageDiagnostic / appendAssistantMessageDiagnostic）。
        diagnostics=[create_assistant_message_diagnostic("provider_error", error)],
        timestamp=now_ms(),
    )


def extract_text(content_blocks: list[ContentBlock]) -> str:
    """
    从 ContentBlock

    提取纯文本。

    会提取：

    TextContent

    ThinkingContent

    并使用换行拼接。

    主要用于：

    日志

    调试

    Provider 转换
    """

    parts: list[str] = []
    for block in content_blocks:
        if block["type"] == "text":
            parts.append(block["text"])
        elif block["type"] == "thinking":
            parts.append(block["thinking"])
    return "\n".join(parts)


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """解析流式累积的 tool call arguments JSON。

    对应 TS 的 parseStreamingJson / repairJson 职责：
    解析失败返回错误占位，避免拖垮整个流。
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": "Invalid JSON arguments"}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
