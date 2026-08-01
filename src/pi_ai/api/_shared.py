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

from typing import Any

from .._types import (
    AssistantMessage,
    ContentBlock,
    Message,
    Model,
    Tool,
    ToolCallContent,
    Usage,
)


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
            result.append({"role": "system", "content": msg["content"]})

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
            content = msg["content"]
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
                                "url": f"data:{block.get('mediaType', 'image/png')};base64,{block['data']}"
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
            oai_msg: dict[str, Any] = {"role": "assistant", "content": None}
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for block in msg["content"]:
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "toolCall":
                    tool_calls.append({
                        "id": block["toolCallId"],
                        "type": "function",
                        "function": {
                            "name": block["toolName"],
                            "arguments": block["args"],
                        },
                    })
                elif block["type"] == "thinking":
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
            content_str = ""
            for block in msg["content"]:
                if block["type"] == "text":
                    content_str += block["text"]

            result.append({
                "role": "tool",
                "tool_call_id": msg["toolCallId"],
                "content": content_str,
            })

    return result


def to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
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
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


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
        cacheRead=0,
        cacheWrite=0,
        totalTokens=0,
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
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
        stopReason="error",
        errorMessage=str(error),
        timestamp=0,
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


def accumulate_tool_calls(
    content: list[ContentBlock],
    index: int | None,
    delta_id: str | None,
    delta_name: str | None,
    delta_args: str | None,
) -> tuple[int | None, str | None]:
    """
    合并 Tool Call 增量。

    OpenAI Streaming

    不会一次返回完整 Tool Call。

    例如：

    Chunk1

    name="search"

    Chunk2

    args="{"

    Chunk3

    args="\"query\""

    Chunk4

    args="}"

    因此需要不断拼接。

    最终得到：

    ToolCallContent
    """

    tool_name: str | None = None

    # 新 Tool Call。
    #
    # 如果不存在，
    #
    # 创建新的 ToolCallContent。
    if delta_id is not None:
        # Find existing block with matching id or create new slot
        found = False
        for i, block in enumerate(content):

            # 查找已有 Tool Call。
            #
            # 如果已经创建，
            #
            # 后续 Chunk
            #
            # 会继续写入。
            if block["type"] == "toolCall" and block["toolCallId"] == delta_id:
                index = i
                found = True
                break

            # 第一次出现 Tool Call。
            #
            # 创建一个空 Block，
            #
            # 后续不断填充。
        if not found:
            content.append(ToolCallContent(
                type="toolCall",
                toolCallId=delta_id,
                toolName="",
                args="",
            ))
            index = len(content) - 1

    if index is not None and index < len(content):
        block = content[index]
        if block["type"] == "toolCall":
            if delta_name:

                # 更新 Tool Name。
                block["toolName"] = delta_name
                tool_name = delta_name

            if delta_args:

                # 拼接 Arguments。
                #
                # OpenAI Arguments
                #
                # 会分多次返回。
                block["args"] += delta_args
                tool_name = block["toolName"] or tool_name

    return index, tool_name
