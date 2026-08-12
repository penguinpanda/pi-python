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

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import httpx
from openai import AsyncOpenAI

from ..utils._background import track_background_task
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.cost import calculate_cost
from ..types import (
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
from ..types.common import ModelThinkingLevel
from ._shared import (
    build_error_message,
    empty_usage,
    parse_tool_arguments,
    to_openai_messages,
    to_openai_tools,
)
from .simple_options import clamp_max_tokens_to_context
from .transform_messages import normalize_tool_call_id, transform_messages
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .compat_runtime import (
    compat_value,
    max_tokens_field,
    supports_long_cache_retention,
    supports_reasoning_effort,
    supports_strict_mode,
    thinking_format,
)
from ..utils.prompt_cache import (
    clamp_openai_prompt_cache_key,
    resolve_cache_retention,
)


# 推理阶段至少为回答保留的 token 数（对齐 TS simple-options.ts MIN_ANSWER_TOKENS）。
MIN_ANSWER_TOKENS = 1024


def _parse_chunk_usage(raw_usage: Any, model: Model) -> Usage:
    """解析 OpenAI-compatible 流式 usage（对齐 TS parseChunkUsage）。

    - cache_read 优先取 ``prompt_tokens_details.cached_tokens``；缺失时回退到
      DeepSeek 返回的顶层 ``prompt_cache_hit_tokens``
    - cache_write 取 ``prompt_tokens_details.cache_write_tokens``（OpenRouter）
    - input = prompt_tokens - cache_read - cache_write，避免缓存命中 token
      同时按输入价与缓存价双重计费
    - total_tokens 重算为 input + output + cache_read + cache_write
    """
    prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
    details = getattr(raw_usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None) if details is not None else None
    cache_read = (
        cached_tokens
        if cached_tokens is not None
        else int(getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0)
    )
    cache_write = int(getattr(details, "cache_write_tokens", 0) or 0) if details is not None else 0
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    output_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
    completion_details = getattr(raw_usage, "completion_tokens_details", None)
    reasoning = (
        int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        if completion_details is not None
        else 0
    )
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )
    if reasoning:
        usage["reasoning"] = reasoning
    calculate_cost(model, usage)
    return usage


def _create_client(
    api_key: str,
    base_url: str,
    timeout: float = 120.0,
    max_retries: int = 2,
    headers: dict[str, str | None] | None = None,
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

    kwargs: dict[str, Any] = dict(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout),
        max_retries=max_retries,
    )
    if headers:
        kwargs["default_headers"] = {k: v for k, v in headers.items() if v is not None}
    return AsyncOpenAI(**kwargs)


async def chat_completions_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str,
    options: StreamOptions | None = None,
    tool_call_id_normalizer: Callable[[str, Model, AssistantMessage], str] | None = None,
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
            request_headers = dict(opts.get("headers") or {})
            if model.provider == "github-copilot":
                # 动态头（对齐 TS buildCopilotDynamicHeaders）：无这些头
                # Copilot 请求（尤其图片）可能被拒。
                request_headers.update(
                    build_copilot_dynamic_headers(
                        messages=context.messages,
                        has_images=has_copilot_vision_input(context.messages),
                    )
                )
            client = _create_client(
                api_key,
                base_url,
                timeout=timeout_ms / 1000.0 if timeout_ms else 120.0,
                max_retries=opts.get("max_retries", 2),
                headers=request_headers or None,
            )

            # 跨 Provider 消息规范化。
            #
            # 图片降级 / thinking 块 / 工具调用 ID 规范化 /
            # 孤立 tool call 合成错误结果。
            transformed_messages = transform_messages(
                context.messages,
                model,
                tool_call_id_normalizer or normalize_tool_call_id,
            )

            # 将规范化后的 SDK Message
            #
            # 转换成 OpenAI Message。
            messages = to_openai_messages(transformed_messages, model)

            # Tool 定义转换为 OpenAI Tool Schema。
            tools = (
                to_openai_tools(
                    context.tools,
                    supports_strict_mode=supports_strict_mode(model),
                )
                if context.tools
                else None
            )

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
            kwargs[max_tokens_field(model)] = clamp_max_tokens_to_context(
                model,
                context,
                requested if requested is not None else model.max_tokens,
            )

            # 提示缓存（Prompt Cache，对齐 TS buildParams）：
            #   prompt_cache_key：官方 OpenAI 且非 none，或 long 且支持长缓存时发送
            #     （session_id 截断到 64 字符）
            #   prompt_cache_retention：long 且支持长缓存时发送 "24h"
            cache_retention = resolve_cache_retention(opts.get("cache_retention"), opts.get("env"))
            supports_long = supports_long_cache_retention(model)
            if ("api.openai.com" in base_url and cache_retention != "none") or (
                cache_retention == "long" and supports_long
            ):
                kwargs["prompt_cache_key"] = clamp_openai_prompt_cache_key(opts.get("session_id"))
            if cache_retention == "long" and supports_long:
                kwargs["prompt_cache_retention"] = "24h"

            # 推理级别转发（对齐 TS openai-completions.ts buildParams 的
            # thinkingFormat 矩阵：zai / qwen / qwen-chat-template /
            # chat-template / baseten / deepseek / openrouter / ant-ling /
            # together / string-thinking + vLLM thinking_token_budget）。
            reasoning_level = cast(ModelThinkingLevel | None, opts.get("reasoning"))
            thinking_map = model.thinking_level_map or {}
            thinking_fmt = thinking_format(model)

            def _effort(level: ModelThinkingLevel | None):
                if level is None:
                    return None
                mapped = thinking_map.get(level)
                return mapped if mapped is not None else level

            _thinking_on = reasoning_level is not None and reasoning_level != "off"

            if model.reasoning and thinking_fmt == "zai":
                if _thinking_on:
                    kwargs["thinking"] = {"type": "enabled", "clear_thinking": False}
                    if supports_reasoning_effort(model):
                        kwargs["reasoning_effort"] = _effort(reasoning_level)
                else:
                    kwargs["thinking"] = {"type": "disabled"}
            elif model.reasoning and thinking_fmt == "qwen":
                kwargs["enable_thinking"] = _thinking_on
                if _thinking_on and supports_reasoning_effort(model):
                    kwargs["reasoning_effort"] = _effort(reasoning_level)
            elif model.reasoning and thinking_fmt == "qwen-chat-template":
                kwargs["chat_template_kwargs"] = {
                    "enable_thinking": _thinking_on,
                    "preserve_thinking": True,
                }
            elif model.reasoning and thinking_fmt in ("chat-template", "baseten"):
                values: dict[str, Any] = {
                    level: value for level, value in thinking_map.items() if isinstance(value, str)
                }
                if values:
                    key = (
                        "chat_template_kwargs"
                        if thinking_fmt == "chat-template"
                        else "chat_template_args"
                    )
                    kwargs[key] = values
                if thinking_fmt == "baseten" and supports_reasoning_effort(model):
                    mapped_effort = (
                        _effort(reasoning_level) if _thinking_on else thinking_map.get("off")
                    )
                    if isinstance(mapped_effort, str):
                        kwargs["reasoning_effort"] = mapped_effort
            elif thinking_fmt == "deepseek" and model.reasoning:
                if _thinking_on:
                    kwargs["thinking"] = {"type": "enabled"}
                    if supports_reasoning_effort(model):
                        kwargs["reasoning_effort"] = _effort(reasoning_level)
                elif "off" not in thinking_map or thinking_map["off"] is not None:
                    kwargs["thinking"] = {"type": "disabled"}
            elif model.reasoning and thinking_fmt == "openrouter":
                if _thinking_on:
                    kwargs["reasoning"] = {"effort": _effort(reasoning_level)}
                elif "off" not in thinking_map or thinking_map["off"] is not None:
                    kwargs["reasoning"] = {"effort": thinking_map.get("off") or "none"}
            elif model.reasoning and thinking_fmt == "ant-ling" and _thinking_on:
                effort = thinking_map.get(cast(ModelThinkingLevel, reasoning_level))
                if isinstance(effort, str):
                    kwargs["reasoning"] = {"effort": effort}
            elif model.reasoning and thinking_fmt == "together":
                kwargs["reasoning"] = {"enabled": _thinking_on}
                if _thinking_on and supports_reasoning_effort(model):
                    kwargs["reasoning_effort"] = _effort(reasoning_level)
            elif model.reasoning and thinking_fmt == "string-thinking":
                if _thinking_on:
                    kwargs["thinking"] = _effort(reasoning_level)
                elif "off" not in thinking_map or thinking_map["off"] is not None:
                    kwargs["thinking"] = thinking_map.get("off") or "none"
            elif _thinking_on and supports_reasoning_effort(model):
                # OpenAI-style reasoning_effort
                kwargs["reasoning_effort"] = _effort(reasoning_level)
            elif not _thinking_on and model.reasoning and supports_reasoning_effort(model):
                off_value = thinking_map.get("off")
                if isinstance(off_value, str):
                    kwargs["reasoning_effort"] = off_value

            # vLLM thinking_token_budget（对齐 TS）：推理与回答共享 max_tokens，
            # 不设上限时推理阶段可能耗尽全部额度、没有回答与工具调用。
            if (
                compat_value(model, "supportsThinkingTokenBudget", False)
                and _thinking_on
                and model.reasoning
            ):
                budgets: dict[str, int] = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 8192,
                    "high": 16384,
                    **cast(dict[str, int], opts.get("thinking_budgets") or {}),
                }
                ceiling = (
                    kwargs.get("max_tokens")
                    or kwargs.get("max_completion_tokens")
                    or model.max_tokens
                )
                budget = min(
                    budgets.get(cast(ModelThinkingLevel, reasoning_level), 0) or 0,
                    max(0, ceiling - MIN_ANSWER_TOKENS),
                )
                if budget > 0:
                    kwargs["thinking_token_budget"] = budget

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
                    text_block = cast(TextContent, content_blocks[current_index])
                    stream.push(
                        TextEndEvent(
                            type="text_end",
                            content_index=current_index,
                            content=text_block["text"],
                            partial=_partial(),
                        )
                    )
                elif current_kind == "thinking" and current_index is not None:
                    thinking_block = cast(ThinkingContent, content_blocks[current_index])
                    stream.push(
                        ThinkingEndEvent(
                            type="thinking_end",
                            content_index=current_index,
                            content=thinking_block["thinking"],
                            partial=_partial(),
                        )
                    )
                elif current_kind == "toolCall" and current_index is not None:
                    tool_block = cast(ToolCall, content_blocks[current_index])
                    tool_block["raw_arguments"] = current_raw_args
                    tool_block["arguments"] = parse_tool_arguments(current_raw_args)
                    stream.push(
                        ToolCallEndEvent(
                            type="toolcall_end",
                            content_index=current_index,
                            tool_call=tool_block,
                            partial=_partial(),
                        )
                    )
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
                # 流式收尾 chunk 可能只带 usage、没有 choices（如 DashScope
                # 兼容模式），必须在 choices 检查之前处理，否则 usage 会丢失。
                if chunk.usage:
                    usage = _parse_chunk_usage(chunk.usage, model)

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta is None:
                    continue

                # 推理内容增量（DeepSeek reasoning_content）。
                #
                # 与 text 块类似，但独立成 thinking 块；DeepSeek 通常先输出
                # reasoning_content 再输出 content。
                # DeepSeek 用 reasoning_content；ollama/qwen3 的 OpenAI 兼容
                # 流式字段名是 reasoning（openai SDK 作为额外字段保留）。
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if not reasoning_delta:
                    reasoning_delta = getattr(delta, "reasoning", None)
                if reasoning_delta:
                    if current_kind != "thinking":
                        _end_current_block()
                        current_kind = "thinking"
                        content_blocks.append(ThinkingContent(type="thinking", thinking=""))
                        current_index = len(content_blocks) - 1
                        stream.push(
                            ThinkingStartEvent(
                                type="thinking_start",
                                content_index=current_index,
                                partial=_partial(),
                            )
                        )
                    idx = cast(int, current_index)
                    thinking_block = cast(ThinkingContent, content_blocks[idx])
                    thinking_block["thinking"] += reasoning_delta
                    stream.push(
                        ThinkingDeltaEvent(
                            type="thinking_delta",
                            content_index=idx,
                            delta=reasoning_delta,
                            partial=_partial(),
                        )
                    )

                # 文本增量。
                #
                # 块切换：当前块不是文本时先结束上一个块。
                if delta.content:
                    if current_kind != "text":
                        _end_current_block()
                        current_kind = "text"
                        content_blocks.append(TextContent(type="text", text=""))
                        current_index = len(content_blocks) - 1
                        stream.push(
                            TextStartEvent(
                                type="text_start",
                                content_index=current_index,
                                partial=_partial(),
                            )
                        )
                    idx = cast(int, current_index)
                    text_block = cast(TextContent, content_blocks[idx])
                    text_block["text"] += delta.content
                    stream.push(
                        TextDeltaEvent(
                            type="text_delta",
                            content_index=idx,
                            delta=delta.content,
                            partial=_partial(),
                        )
                    )

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
                            content_blocks.append(
                                ToolCall(
                                    type="toolCall",
                                    id=tc_id or "",
                                    name=tc_name or "",
                                    raw_arguments="",
                                    arguments=None,
                                )
                            )
                            current_index = len(content_blocks) - 1
                            stream.push(
                                ToolCallStartEvent(
                                    type="toolcall_start",
                                    content_index=current_index,
                                    partial=_partial(),
                                )
                            )

                        tool_block = cast(ToolCall, content_blocks[cast(int, current_index)])

                        # 工具名称可能延迟到达。
                        if tc_name:
                            tool_block["name"] = tc_name

                        if tc_args:
                            current_raw_args += tc_args
                            stream.push(
                                ToolCallDeltaEvent(
                                    type="toolcall_delta",
                                    content_index=cast(int, current_index),
                                    delta=tc_args,
                                    partial=_partial(),
                                )
                            )

                # Finish reason
                if choice.finish_reason:
                    stop_reason = choice.finish_reason

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
            stream.push(
                {
                    "type": "done",
                    "reason": cast(Any, msg["stop_reason"]),
                    "message": msg,
                }
            )
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
    track_background_task(_run())

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
