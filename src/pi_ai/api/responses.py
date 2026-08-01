"""
OpenAI Responses API 实现。

=========================================================
模块职责
=========================================================

本模块负责调用 OpenAI Responses API，

并将其事件(Event)流转换为 SDK 内部事件流。

整个处理流程如下：

        Context
            │
            ▼
    _to_responses_input()

            │
            ▼
OpenAI Responses API

            │
            ▼
Responses Event Stream

            │
            ▼
转换为 SDK Event

            │
            ▼
AssistantMessageEventStream

与 Chat Completions API 不同，

Responses API 返回的是"事件(Event)"，

例如：

    response.output_text.delta

    response.function_call_arguments.delta

    response.completed

因此本模块本质上是：

    Responses Event

        ↓

    SDK Event

的适配层(Adapter)。
"""

import asyncio
from typing import Any

import httpx
from openai import AsyncOpenAI

from .._event_stream import AssistantMessageEventStream
from .._types import (
    AssistantMessage,
    Context,
    Model,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    Usage,
    ContentBlock,
)
from ._shared import (
    build_error_message,
    empty_usage,
    to_openai_tools,
)


def _create_client(api_key: str, base_url: str = "", timeout: float = 180.0) -> AsyncOpenAI:
    """
    创建 AsyncOpenAI 客户端。

    统一配置：

        • API Key
        • Base URL
        • Timeout
        • Retry

    Responses API 与 Chat Completions API

    共用 AsyncOpenAI 客户端。
    """

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": httpx.Timeout(timeout),
        "max_retries": 2,
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")

    return AsyncOpenAI(**kwargs)


def _to_responses_input(
    messages: list[Any],
    system: str | None,
    model: Model | None = None,
) -> list[dict[str, Any]]:
    """
    将 SDK Message

    转换为 Responses API Input。

    不同于 Chat Completions：

    Responses API

    使用 input 字段，

    格式也有所不同。

    例如：

    SDK

    ↓

    UserMessage

    ↓

    Responses Input Item

    同时负责：

        • System

        • User

        • Assistant History

        • Tool Result

    全部转换。

    Parameters
    ----------
    model
        可选的模型元数据。

        用于按模型能力过滤图片输入。

        当模型不支持 image 时，图片内容会被跳过。

        为 None 时不做过滤（保持向后兼容）。
    """
    items: list[dict[str, Any]] = []

    # System prompt as first item
    if system:
        items.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]

        if role == "system":
            # System Prompt。
            #
            # Responses API
            #
            # 可以在对话任意位置出现
            # System Message。
            items.append({"role": "system", "content": msg["content"]})

        # User Message。
        #
        # 支持：
        #
        # • 文本
        #
        # • 图片
        #
        # 转换成 Responses API
        #
        # input_text
        #
        # input_image。
        elif role == "user":
            content = msg["content"]
            if isinstance(content, str):
                items.append({"role": "user", "content": content})
            else:
                # Multi-part user content
                parts: list[dict[str, Any]] = []
                for block in content:
                    if block["type"] == "text":
                        parts.append({"type": "input_text", "text": block["text"]})
                    elif block["type"] == "image" and (model is None or "image" in (model.input or [])):
                        img: dict[str, Any] = {"type": "input_image"}
                        if block.get("url"):
                            img["image_url"] = block["url"]
                        elif block.get("data"):

                            # Base64 图片
                            #
                            # 转换为 Data URL。
                            img["image_url"] = (
                                f"data:{block.get('mediaType', 'image/png')};base64,{block['data']}"
                            )
                        parts.append(img)
                items.append({"role": "user", "content": parts})

        # Assistant 历史消息。
        #
        # Responses API
        #
        # 使用 output_text。
        elif role == "assistant":
            # Responses API uses "assistant" items for history
            content_parts: list[dict[str, Any]] = []
            for block in msg["content"]:
                if block["type"] == "text":
                    content_parts.append({"type": "output_text", "text": block["text"]})
            items.append({"role": "assistant", "content": content_parts})

        # Tool 调用结果。
        #
        # Responses API
        #
        # 使用 function_call_output。
        elif role == "toolResult":
            # Function call output
            text = ""
            for block in msg["content"]:
                if block["type"] == "text":
                    text += block["text"]
            items.append({
                "type": "function_call_output",
                "call_id": msg["toolCallId"],
                "output": text,
            })

    return items



async def responses_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """
    执行一次 Responses API

    流式请求。

    返回：

    AssistantMessageEventStream。

    该函数立即返回 EventStream，

    真正的网络请求在后台协程中执行。

    调用者可以立刻：

        async for

    监听模型输出。
    """

    # SDK 内部事件流。
    #
    # Responses Event
    #
    # 会转换成这里的 Event。
    stream = AssistantMessageEventStream()
    opts = options or {}

    async def _run() -> None:
        """
        后台协程。

        负责：

        创建 Client

        ↓

        发送请求

        ↓

        读取 Responses Event

        ↓

        转换 SDK Event

        ↓

        生成最终 AssistantMessage
        """

        try:
            client = _create_client(api_key, base_url)
            input_items = _to_responses_input(context.messages, context.system, model)
            tools = to_openai_tools(context.tools) if context.tools else None

            # Responses API 请求参数。
            #
            # 根据用户配置，
            #
            # 动态添加：
            #
            # temperature
            #
            # max_tokens
            #
            # tools。
            kwargs: dict[str, Any] = {
                "model": model.id,
                "input": input_items,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools
            temperature = opts.get("temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
            max_tokens = opts.get("maxTokens")
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            # 发起流式请求。
            #
            # 返回异步 Event Stream。
            response = await client.responses.create(**kwargs)

            # 最终 AssistantMessage.content。
            content_blocks: list[ContentBlock] = [] 

            # 收集模型推理内容。
            #
            # Responses API
            #
            # Thinking 会单独返回。
            reasoning_text = ""

            # 最终回复文本。
            current_text = ""

            # Token 使用统计。
            usage: Usage = empty_usage()
            stop_reason = "end"

            # 当前 Tool Call ID。
            current_call_id: str | None = None

            # 当前 Tool 名称。
            current_call_name: str = ""

            # 当前 Tool 参数。
            #
            # 会持续拼接。
            current_call_args: str = ""

            # 持续读取 Responses Event。
            #
            # 不同事件
            #
            # 分别处理。
            async for event in response:
                event_type = getattr(event, "type", None)

                # 文本增量。
                #
                # 一方面保存，
                #
                # 一方面立即推送 Event。
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    current_text += delta
                    stream.push({"type": "delta", "text": delta})

                # 新 Tool Call。
                #
                # 创建 ToolCall Block。
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", None) == "function_call":
                        current_call_id = getattr(item, "call_id", "")
                        current_call_name = getattr(item, "name", "")
                        current_call_args = ""
                        content_blocks.append(ToolCallContent(
                            type="toolCall",
                            toolCallId=current_call_id or "",
                            toolName="",
                            args="",
                        ))

                # Tool 参数增量。
                #
                # 持续拼接 Arguments。
                elif event_type == "response.function_call_arguments.delta":
                    delta = getattr(event, "delta", "")
                    current_call_args += delta
                    stream.push({
                        "type": "toolCallDelta",
                        "toolCallId": current_call_id or "",
                        "toolName": current_call_name,
                        "argsDelta": delta,
                    })

                # Tool 参数结束。
                #
                # 更新最终 ToolCall Block。
                elif event_type == "response.function_call_arguments.done":
                    # Finalize the tool call args
                    for block in content_blocks:
                        if block["type"] == "toolCall" and block["toolCallId"] == current_call_id:
                            block["toolName"] = current_call_name
                            block["args"] = current_call_args
                            break

                # 模型推理内容。
                #
                # Responses API
                #
                # 会独立发送。
                elif event_type == "response.reasoning_summary_part.added":
                    summary = getattr(event, "part", None)
                    if summary and getattr(summary, "type", None) == "summary_text":
                        text = getattr(summary, "text", "")
                        reasoning_text += text
                        stream.push({"type": "thinkingDelta", "thinking": text})

                elif event_type == "response.reasoning_text.delta":
                    delta = getattr(event, "delta", "")
                    reasoning_text += delta
                    stream.push({"type": "thinkingDelta", "thinking": delta})

                # 整个 Responses 请求结束。
                #
                # 获取：
                #
                # 输出文本
                #
                # Usage。
                elif event_type == "response.completed":
                    resp = getattr(event, "response", None)
                    if resp:
                        # Extract output text
                        output_text = getattr(resp, "output_text", "")
                        if output_text:
                            current_text = output_text

                        # Extract usage
                        if hasattr(resp, "usage") and resp.usage:
                            usage = Usage(
                                input=resp.usage.input_tokens or 0,
                                output=resp.usage.output_tokens or 0,
                                cacheRead=0,
                                cacheWrite=0,
                                totalTokens=resp.usage.total_tokens or 0,
                            )

            if reasoning_text:
                    content_blocks.append(ThinkingContent(
                        type="thinking",
                        thinking=reasoning_text,
                        signature=None,
                    ))
            # 构造最终 ContentBlock。
            if current_text:
                content_blocks.append(TextContent(type="text", text=current_text))

            msg = AssistantMessage(
                role="assistant",
                content=content_blocks,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=usage,
                stopReason=stop_reason,
                errorMessage=None,
                timestamp=0,
            )
            stream.push({"type": "done", "message": msg})
            # stream.end(msg)

        except asyncio.CancelledError:
            # 让 await stream.result() 抛出取消异常，而不是永久挂起。
            stream.error(asyncio.CancelledError())
            raise

        except Exception as exc:
            err_msg = build_error_message(model, exc)
            stream.push({"type": "error", "reason": "error", "error": err_msg})
            # stream.end(err_msg)

    asyncio.create_task(_run())
    return stream
