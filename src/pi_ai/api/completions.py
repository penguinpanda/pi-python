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
    to_openai_messages,
    to_openai_tools,
)
from .simple_options import clamp_max_tokens_to_context
from .transform_messages import normalize_tool_call_id, transform_messages


def _create_client(
    api_key: str,
    base_url: str,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """
    创建 AsyncOpenAI 客户端。

    封装客户端创建逻辑，统一配置：

        - API Key
        - Base URL
        - Timeout
        - Retry（默认 2，可从 StreamOptions.max_retries 覆盖）

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
        max_retries=max_retries,
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
            #
            # 重试参数从 StreamOptions 读取（缺省保持 SDK 默认 2）。
            # retry-after / retry-after-ms / x-should-retry / 408/409/429/5xx
            # 由 openai SDK 内置处理，指数退避封顶 60s。
            # （max_retry_delay_ms 暂不生效：SDK 客户端不接受该参数。）
            timeout_ms = opts.get("timeout_ms")
            client = _create_client(
                api_key,
                base_url,
                timeout=timeout_ms / 1000.0 if timeout_ms else 120.0,
                max_retries=opts.get("max_retries", 2),
            )

            # 跨 Provider 消息规范化。
            #
            # 图片降级 / thinking 块 / 工具调用 ID 规范化 /
            # 孤立 tool call 合成错误结果。
            transformed_messages = transform_messages(
                context.messages, model, normalize_tool_call_id
            )

            # 将规范化后的 SDK Message
            #
            # 转换成 OpenAI Message。
            messages = to_openai_messages(transformed_messages, model)

            # Tool 定义转换为 OpenAI Tool Schema。
            tools = to_openai_tools(context.tools) if context.tools else None

            # Chat Completions API
            #
            # System Prompt 作为第一条 message。
            if context.system_prompt:
                messages.insert(0, {"role": "system", "content": context.system_prompt})

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
            # max_tokens 收敛到模型上下文窗口内（对齐 TS buildBaseOptions）：
            # 未指定时使用模型默认 max_tokens，始终发送收敛后的值。
            requested = opts.get("max_tokens")
            kwargs["max_tokens"] = clamp_max_tokens_to_context(
                model,
                context,
                requested if requested is not None else model.max_tokens,
            )

            # 发起流式请求。
            #
            # 返回的是异步可迭代对象。
            response = await client.chat.completions.create(**kwargs)

            # 最终 AssistantMessage.content。
            content_blocks: list[ContentBlock] = []

            # 当前正在累积的内容块（text / toolCall）。
            current_index: int | None = None
            current_kind: str | None = None

            # 当前 toolCall 的流式状态。
            current_tool_id: str | None = None
            current_raw_args: str = ""

            # Token 使用统计。
            usage: Usage = empty_usage()

            # 停止标识。
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
                    block = cast(TextContent, content_blocks[current_index])
                    stream.push(TextEndEvent(
                        type="text_end",
                        content_index=current_index,
                        content=block["text"],
                        partial=_partial(),
                    ))
                elif current_kind == "toolCall" and current_index is not None:
                    block = cast(ToolCall, content_blocks[current_index])
                    block["raw_arguments"] = current_raw_args
                    block["arguments"] = parse_tool_arguments(current_raw_args)
                    stream.push(ToolCallEndEvent(
                        type="toolcall_end",
                        content_index=current_index,
                        tool_call=block,
                        partial=_partial(),
                    ))
                current_kind = None
                current_index = None
                current_tool_id = None

            # 流开始事件。
            stream.push(StartEvent(type="start", partial=_partial()))

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
                # 块切换：当前块不是文本时先结束上一个块。
                if delta.content:
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
                    idx = cast(int, current_index)
                    block = cast(TextContent, content_blocks[idx])
                    block["text"] += delta.content
                    stream.push(TextDeltaEvent(
                        type="text_delta",
                        content_index=idx,
                        delta=delta.content,
                        partial=_partial(),
                    ))

                # Tool Calling 增量。
                #
                # Tool 参数可能被拆分成多个 Chunk，
                #
                # 因此需要不断拼接原始字符串，
                # 块结束时再解析为 JSON。
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_id = tc.id or current_tool_id
                        tc_name = tc.function.name if tc.function else None
                        tc_args = tc.function.arguments if tc.function else None

                        # 块切换：当前块不是 toolCall，或出现了新的 toolCall id。
                        if current_kind != "toolCall" or (tc_id and tc_id != current_tool_id):
                            _end_current_block()
                            current_kind = "toolCall"
                            current_tool_id = tc_id
                            current_raw_args = ""
                            content_blocks.append(ToolCall(
                                type="toolCall",
                                id=tc_id or "",
                                name=tc_name or "",
                                raw_arguments="",
                                arguments=None,
                            ))
                            current_index = len(content_blocks) - 1
                            stream.push(ToolCallStartEvent(
                                type="toolcall_start",
                                content_index=current_index,
                                partial=_partial(),
                            ))

                        block = cast(ToolCall, content_blocks[cast(int, current_index)])

                        # 工具名称可能延迟到达。
                        if tc_name:
                            block["name"] = tc_name

                        if tc_args:
                            current_raw_args += tc_args
                            stream.push(ToolCallDeltaEvent(
                                type="toolcall_delta",
                                content_index=cast(int, current_index),
                                delta=tc_args,
                                partial=_partial(),
                            ))

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
                        cache_read=cached_tokens,
                        cache_write=0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
                    )

            # 所有 Chunk 已处理完成，
            #
            # 结束最后一个块（若有）。
            _end_current_block()

            # 构造最终 AssistantMessage。
            msg = AssistantMessage(
                role="assistant",
                content=content_blocks,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=usage,
                stop_reason=_map_stop_reason(stop_reason),
                error_message=None,
                timestamp=now_ms(),
            )
            # reason 取映射后的 stop_reason。
            #
            # content_filter 等映射为 "error" 的罕见情况
            # 仍以 done 事件结束（保持既有行为）。
            stream.push({
                "type": "done",
                "reason": cast(Any, msg["stop_reason"]),
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

        stop / end          → stop
        length              → length
        tool_calls          → tool_call
        function_call       → tool_call
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
        "tool_calls": "tool_call",
        "function_call": "tool_call",
        "content_filter": "error",
        "network_error": "error",
    }
    return cast(StopReason, mapping.get(reason, "error"))
