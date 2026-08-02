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
import json
from typing import Any, cast

import httpx
from openai import AsyncOpenAI

from ..utils._event_stream import AssistantMessageEventStream
from .._types import (
    AssistantMessage,
    ContentBlock,
    Context,
    Model,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    now_ms,
)
from ._shared import (
    build_error_message,
    empty_usage,
    parse_tool_arguments,
    to_openai_tools,
)
from .transform_messages import transform_messages


def _create_client(
    api_key: str,
    base_url: str = "",
    timeout: float = 180.0,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """
    创建 AsyncOpenAI 客户端。

    统一配置：

        • API Key
        • Base URL
        • Timeout
        • Retry（默认 2，可从 StreamOptions.max_retries 覆盖）

    Responses API 与 Chat Completions API

    共用 AsyncOpenAI 客户端。
    """

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": httpx.Timeout(timeout),
        "max_retries": max_retries,
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
                                f"data:{block.get('mime_type', 'image/png')};base64,{block['data']}"
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
                elif block["type"] == "toolCall":
                    # 工具调用历史 → function_call item。
                    #
                    # transform 合成的孤立 toolResult 在 Responses 侧
                    # 会转为 function_call_output；必须有对应的
                    # function_call 历史才合法。
                    content_parts.append({
                        "type": "function_call",
                        "call_id": block["id"],
                        "name": block["name"],
                        "arguments": json.dumps(
                            block["arguments"] if block["arguments"] is not None else {},
                            ensure_ascii=False,
                        ),
                    })
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
                "call_id": msg["tool_call_id"],
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
            # 创建 OpenAI SDK 客户端。
            #
            # 重试参数从 StreamOptions 读取（缺省保持 SDK 默认 2）。
            # retry-after / retry-after-ms / x-should-retry / 408/409/429/5xx
            # 由 openai SDK 内置处理，指数退避封顶 60s。
            # （max_retry_delay_ms 暂不生效：SDK 客户端不接受该参数。）
            timeout_ms = opts.get("timeout_ms")
            client = _create_client(
                api_key,
                base_url,
                timeout=timeout_ms / 1000.0 if timeout_ms else 180.0,
                max_retries=opts.get("max_retries", 2),
            )
            # 跨 Provider 消息规范化。
            #
            # 图片降级 / thinking 块 / 工具调用 ID 规范化 /
            # 孤立 tool call 合成错误结果。
            transformed_messages = transform_messages(context.messages, model)
            input_items = _to_responses_input(
                transformed_messages, context.system_prompt, model
            )
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
            max_tokens = opts.get("max_tokens")
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            # 发起流式请求。
            #
            # 返回异步 Event Stream。
            response = await client.responses.create(**kwargs)

            # 最终 AssistantMessage.content。
            content_blocks: list[ContentBlock] = []

            # 当前正在累积的内容块（text / thinking / toolCall）。
            current_index: int | None = None
            current_kind: str | None = None

            # 当前 toolCall 的流式状态。
            current_tool_id: str = ""
            current_tool_name: str = ""
            current_raw_args: str = ""

            # Token 使用统计。
            usage: Usage = empty_usage()
            stop_reason: StopReason = "stop"

            def _partial() -> AssistantMessage:
                """构造当前累积状态的 partial AssistantMessage 快照。"""
                return AssistantMessage(
                    role="assistant",
                    content=cast(list[ContentBlock], [dict(block) for block in content_blocks]),
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    usage=usage,
                    stop_reason="pending",
                    timestamp=now_ms(),
                )

            def _end_current_block() -> None:
                """结束当前累积块并发射对应的 *_end 事件。"""
                nonlocal current_kind, current_index, current_tool_id
                if current_kind == "text" and current_index is not None:
                    block = content_blocks[current_index]
                    stream.push(TextEndEvent(
                        type="text_end",
                        content_index=current_index,
                        content=block["text"],
                        partial=_partial(),
                    ))
                elif current_kind == "thinking" and current_index is not None:
                    block = content_blocks[current_index]
                    stream.push(ThinkingEndEvent(
                        type="thinking_end",
                        content_index=current_index,
                        content=block["thinking"],
                        partial=_partial(),
                    ))
                elif current_kind == "toolCall" and current_index is not None:
                    block = content_blocks[current_index]
                    block["raw_arguments"] = current_raw_args
                    block["arguments"] = parse_tool_arguments(current_raw_args)
                    stream.push(ToolCallEndEvent(
                        type="toolcall_end",
                        content_index=current_index,
                        tool_call=cast(ToolCall, block),
                        partial=_partial(),
                    ))
                current_kind = None
                current_index = None
                current_tool_id = ""

            def _begin_text() -> int:
                """确保当前块为文本块，返回 content_index。"""
                nonlocal current_kind, current_index
                if current_kind != "text":
                    _end_current_block()
                    current_kind = "text"
                    content_blocks.append(TextContent(type="text", text=""))
                    current_index = len(content_blocks) - 1
                    stream.push(TextStartEvent(
                        type="text_start",
                        content_index=current_index,
                        partial=_partial(),
                    ))
                return current_index  # type: ignore[return-value]

            def _begin_thinking() -> int:
                """确保当前块为思考块，返回 content_index。"""
                nonlocal current_kind, current_index
                if current_kind != "thinking":
                    _end_current_block()
                    current_kind = "thinking"
                    content_blocks.append(ThinkingContent(type="thinking", thinking=""))
                    current_index = len(content_blocks) - 1
                    stream.push(ThinkingStartEvent(
                        type="thinking_start",
                        content_index=current_index,
                        partial=_partial(),
                    ))
                return current_index  # type: ignore[return-value]

            # 流开始事件。
            stream.push(StartEvent(type="start", partial=_partial()))

            # 持续读取 Responses Event。
            #
            # 不同事件
            #
            # 分别处理。
            async for event in response:
                event_type = getattr(event, "type", None)

                # 文本增量。
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    idx = _begin_text()
                    content_blocks[idx]["text"] += delta
                    stream.push(TextDeltaEvent(
                        type="text_delta",
                        content_index=idx,
                        delta=delta,
                        partial=_partial(),
                    ))

                # 新 Tool Call。
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", None) == "function_call":
                        _end_current_block()
                        current_kind = "toolCall"
                        current_tool_id = getattr(item, "call_id", "") or ""
                        current_tool_name = getattr(item, "name", "") or ""
                        current_raw_args = ""
                        content_blocks.append(ToolCall(
                            type="toolCall",
                            id=current_tool_id,
                            name=current_tool_name,
                            raw_arguments="",
                            arguments=None,
                        ))
                        current_index = len(content_blocks) - 1
                        stream.push(ToolCallStartEvent(
                            type="toolcall_start",
                            content_index=current_index,
                            partial=_partial(),
                        ))

                # Tool 参数增量。
                elif event_type == "response.function_call_arguments.delta":
                    delta = getattr(event, "delta", "")
                    current_raw_args += delta
                    stream.push(ToolCallDeltaEvent(
                        type="toolcall_delta",
                        content_index=current_index or 0,
                        delta=delta,
                        partial=_partial(),
                    ))

                # Tool 参数结束。
                elif event_type == "response.function_call_arguments.done":
                    _end_current_block()

                # 模型推理内容。
                elif event_type == "response.reasoning_summary_part.added":
                    summary = getattr(event, "part", None)
                    if summary and getattr(summary, "type", None) == "summary_text":
                        text = getattr(summary, "text", "")
                        idx = _begin_thinking()
                        content_blocks[idx]["thinking"] += text
                        stream.push(ThinkingDeltaEvent(
                            type="thinking_delta",
                            content_index=idx,
                            delta=text,
                            partial=_partial(),
                        ))

                elif event_type == "response.reasoning_text.delta":
                    delta = getattr(event, "delta", "")
                    idx = _begin_thinking()
                    content_blocks[idx]["thinking"] += delta
                    stream.push(ThinkingDeltaEvent(
                        type="thinking_delta",
                        content_index=idx,
                        delta=delta,
                        partial=_partial(),
                    ))

                # 整个 Responses 请求结束。
                elif event_type == "response.completed":
                    resp = getattr(event, "response", None)
                    if resp:
                        # 权威输出文本：覆盖实时累积的 text 块；
                        # 若尚未有 text 块（如仅 completed 事件），则新建。
                        output_text = getattr(resp, "output_text", "")
                        if output_text:
                            if current_kind == "text" and current_index is not None:
                                content_blocks[current_index]["text"] = output_text
                            else:
                                _end_current_block()
                                current_kind = "text"
                                content_blocks.append(TextContent(type="text", text=output_text))
                                current_index = len(content_blocks) - 1
                                stream.push(TextStartEvent(
                                    type="text_start",
                                    content_index=current_index,
                                    partial=_partial(),
                                ))
                                stream.push(TextEndEvent(
                                    type="text_end",
                                    content_index=current_index,
                                    content=output_text,
                                    partial=_partial(),
                                ))

                        # Extract usage
                        if hasattr(resp, "usage") and resp.usage:
                            usage = Usage(
                                input=resp.usage.input_tokens or 0,
                                output=resp.usage.output_tokens or 0,
                                cache_read=0,
                                cache_write=0,
                                total_tokens=resp.usage.total_tokens or 0,
                                cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
                            )

            # 所有事件已处理完成，
            #
            # 结束最后一个块（若有）。
            _end_current_block()

            msg = AssistantMessage(
                role="assistant",
                content=content_blocks,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=usage,
                stop_reason=stop_reason,
                error_message=None,
                timestamp=now_ms(),
            )
            # reason 与 stop_reason 一致；Responses API 无独立 error 映射，
            # 未知情况仍以 done 事件结束（保持既有行为）。
            stream.push({
                "type": "done",
                "reason": cast(Any, stop_reason),
                "message": msg,
            })
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
