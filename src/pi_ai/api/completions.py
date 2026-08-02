"""
OpenAI Chat Completions API 实现。

=========================================================
模块职责
=========================================================

本模块负责：

    ① 调用 OpenAI Chat Completions API

    ② 将 OpenAI Streaming Chunk
       转换为 SDK 内部事件(Event)

    ③ 构造最终 AssistantMessage

    ④ 返回 AssistantMessageEventStream

整个流程如下：

        Context
            │
            ▼
to_openai_messages()

            │
            ▼
OpenAI Streaming API

            │
            ▼
Streaming Chunk

            │
            ▼
转换成 SDK Event

            │
            ▼
AssistantMessageEventStream

Provider 不需要关心 OpenAI SDK 的数据结构，
统一消费 EventStream 即可。
"""

import asyncio
from typing import Any, cast

import httpx
from openai import AsyncOpenAI

from .._event_stream import AssistantMessageEventStream
from .._types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    StreamOptions,
    TextContent,
    Usage,
)
from ._shared import (
    accumulate_tool_calls,
    build_error_message,
    empty_usage,
    to_openai_messages,
    to_openai_tools,
)


def _create_client(api_key: str, base_url: str, timeout: float = 120.0) -> AsyncOpenAI:
    """
    创建 AsyncOpenAI 客户端。

    封装客户端创建逻辑，统一配置：

        - API Key
        - Base URL
        - Timeout
        - Retry

    例如：

    官方：

        https://api.openai.com/v1

    兼容接口：

        https://api.deepseek.com

        https://openrouter.ai/api/v1

    都可以通过 base_url 指定。
    """
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout),
        max_retries=2,
    )


async def chat_completions_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """
    执行一次 Chat Completions 流式请求。

    返回：

        AssistantMessageEventStream

    注意：

    该函数不会等待模型返回。

    而是：

    ① 创建 EventStream

    ② 启动后台任务

    ③ 立即返回 EventStream

    真正的网络请求在后台协程 _run() 中执行。

    因此调用者可以立刻：

        async for event in stream

    开始监听模型输出。
    """

    # SDK 内部事件流。
    #
    # OpenAI 返回的数据都会被转换成
    # Event 后推送到这里。
    stream = AssistantMessageEventStream()

    # 用户可选参数。
    #
    # 如果没有提供，
    # 使用空字典方便后续 get()。
    opts = options or {}

    async def _run() -> None:
        """
        后台协程。

        负责：

            创建 Client

                ↓

            调用 OpenAI API

                ↓

            持续读取 Streaming Chunk

                ↓

            转换为 Event

                ↓

            构造最终 AssistantMessage
        """
        try:
            # 创建 OpenAI SDK 客户端。
            client = _create_client(api_key, base_url)

            # 将 SDK Message
            #
            # 转换成 OpenAI Message。
            messages = to_openai_messages(context.messages, model)

            # Tool 定义转换为 OpenAI Tool Schema。
            tools = to_openai_tools(context.tools) if context.tools else None

            # Chat Completions API
            #
            # System Prompt 作为第一条 message。
            if context.system:
                messages.insert(0, {"role": "system", "content": context.system})

            # OpenAI Chat Completions 参数。
            #
            # 根据用户 Options
            #
            # 动态添加：
            #
            # temperature
            # max_tokens
            # tools
            kwargs: dict[str, Any] = {
                "model": model.id,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
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
            # 返回的是异步可迭代对象。
            response = await client.chat.completions.create(**kwargs)

            # 最终 AssistantMessage.content。
            content_blocks: list = []

            # Tool Call 顺序编号。
            tool_index: int | None = None

            # index -> toolCallId
            #
            # OpenAI Tool Call
            #
            # 会分多次返回，
            # 因此需要保存映射。
            tool_call_ids: dict[int, str] = {}  # index -> toolCallId

            # 当前 Tool 名称。
            current_tool_name: str | None = None

            # Token 使用统计。
            usage: Usage = empty_usage()

            # 停止标识。
            stop_reason: StopReason = "stop"

            # 持续读取 OpenAI Streaming Chunk。
            #
            # 每一个 chunk
            #
            # 都可能包含：
            #
            # 文本
            # Tool Call
            # Usage
            # Finish Reason
            async for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta is None:
                    continue

                # 文本增量。
                #
                # OpenAI 每次返回一小段文本。
                #
                # 一方面保存到最终消息，
                #
                # 一方面立即推送 Event。
                if delta.content:
                    content_blocks.append(TextContent(type="text", text=delta.content))
                    stream.push({"type": "delta", "text": delta.content})

                # Tool Calling 增量。
                #
                # Tool 参数可能被拆分成多个 Chunk，
                #
                # 因此需要不断拼接。
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else tool_index
                        # Track tool call ID
                        if tc.id:
                            tool_call_ids[idx or 0] = tc.id

                        # 将 Tool Call 增量
                        #
                        # 合并到 content_blocks。
                        tool_index, tool_name = accumulate_tool_calls(
                            content=content_blocks,
                            index=idx,
                            delta_id=tc.id,
                            delta_name=tc.function.name if tc.function else None,
                            delta_args=tc.function.arguments if tc.function else None,
                        )
                        if tool_name:
                            current_tool_name = tool_name
                        if tc.function and tc.function.arguments:
                            stream.push({
                                "type": "toolCallDelta",
                                "toolCallId": tool_call_ids.get(idx or 0, ""),
                                "toolName": current_tool_name or "",
                                "argsDelta": tc.function.arguments,
                            })

                # Finish reason
                if choice.finish_reason:
                    stop_reason = choice.finish_reason

                # Streaming 最后一个 Chunk
                #
                # 包含 Token Usage。
                if chunk.usage:
                    cached_tokens = getattr(getattr(chunk.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                    usage = Usage(
                        input=chunk.usage.prompt_tokens or 0,
                        output=chunk.usage.completion_tokens or 0,
                        cacheRead=cached_tokens,
                        cacheWrite=0,
                        totalTokens=chunk.usage.total_tokens or 0,
                    )

            # 所有 Chunk 已处理完成，
            #
            # 构造最终 AssistantMessage。
            msg = AssistantMessage(
                role="assistant",
                content=content_blocks,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=usage,
                stopReason=_map_stop_reason(stop_reason),
                errorMessage=None,
                timestamp=0,
            )
            # reason 取映射后的 stopReason。
            #
            # content_filter 等映射为 "error" 的罕见情况
            # 仍以 done 事件结束（保持既有行为）。
            stream.push({
                "type": "done",
                "reason": cast(Any, msg["stopReason"]),
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

    # 后台启动网络请求。
    #
    # 不等待完成，
    #
    # 立即返回 EventStream。
    asyncio.create_task(_run())

    return stream


def _map_stop_reason(reason: str) -> StopReason:
    """
    将 OpenAI Finish Reason

    转换为 SDK 内部 Stop Reason。

    与 TS mapStopReason() 对齐：

        stop / end          → stop
        length              → length
        tool_calls          → toolUse
        function_call       → toolUse
        content_filter      → error
        network_error       → error
        空                   → stop（等价 TS 的 null）
        其他                 → error
    """
    if not reason:
        return "stop"
    mapping = {
        "stop": "stop",
        "end": "stop",
        "length": "length",
        "tool_calls": "toolUse",
        "function_call": "toolUse",
        "content_filter": "error",
        "network_error": "error",
    }
    return mapping.get(reason, "error")
