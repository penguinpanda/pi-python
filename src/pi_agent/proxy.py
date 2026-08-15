"""Proxy 流函数（Phase 5.1）。

对齐 TS `packages/agent/src/proxy.ts`：把 LLM 调用路由经过服务器中转，
服务器负责认证与 provider 代理，客户端通过 SSE 接收事件。

带宽优化：服务器剥离 `partial` 字段，客户端基于事件流本地重建 partial
消息（文本拼接、thinking 拼接、toolCall 参数流式解析）。

用法（作为 Agent 的 stream_fn）：

    from pi_agent.proxy import stream_proxy
    agent = Agent(AgentOptions(
        model=model,
        stream_fn=lambda model, context, options: stream_proxy(model, context, {
            **(options or {}),
            "authToken": token,
            "proxyUrl": "https://genai.example.com",
        }),
    ))
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Literal, cast

import httpx

from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ContentBlock,
    Context,
    Model,
    ToolCall,
    Usage,
    now_ms,
)
from pi_ai.utils._event_stream import AssistantMessageEventStream
from pi_ai.utils.partial_json import partial_json


class ProxyMessageEventStream(AssistantMessageEventStream):
    """代理消息事件流：done/error 即结束。"""


def _empty_usage() -> Usage:
    return {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def _snake_to_camel(key: str) -> str:
    head, *tail = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in tail if word)


def _to_jsonable(value: Any) -> Any:
    """把 Python 载荷转为 proxy 线协议 JSON（camelCase，对齐 TS）。"""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            _snake_to_camel(field.name): _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            _snake_to_camel(str(key)) if "_" in str(key) else str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def build_proxy_request_options(options: dict[str, Any]) -> dict[str, Any]:
    """挑选可序列化的流选项并映射为 TS 线协议键（buildProxyRequestOptions）。"""
    keys = (
        ("temperature", "temperature"),
        ("sampling_params", "samplingParams"),
        ("max_tokens", "maxTokens"),
        ("reasoning", "reasoning"),
        ("cache_retention", "cacheRetention"),
        ("session_id", "sessionId"),
        ("headers", "headers"),
        ("metadata", "metadata"),
        ("transport", "transport"),
        ("thinking_budgets", "thinkingBudgets"),
        ("max_retry_delay_ms", "maxRetryDelayMs"),
    )
    return {
        wire_key: _to_jsonable(options[key])
        for key, wire_key in keys
        if key in options and options[key] is not None
    }


def process_proxy_event(
    proxy_event: dict[str, Any],
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """根据代理事件重建 partial 消息，返回完整事件（对齐 TS processProxyEvent）。"""
    event_type = proxy_event.get("type")
    content = partial.setdefault("content", [])

    def _set_content(index: int, block: dict[str, Any]) -> None:
        block_c = cast(ContentBlock, block)
        while len(content) <= index:
            content.append(block_c)
        content[index] = block_c

    if event_type == "start":
        return {"type": "start", "partial": partial}

    if event_type == "text_start":
        _set_content(proxy_event["contentIndex"], {"type": "text", "text": ""})
        return {
            "type": "text_start",
            "content_index": proxy_event["contentIndex"],
            "partial": partial,
        }

    if event_type == "text_delta":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError("Received text_delta for non-text content")
        block["text"] += proxy_event["delta"]
        return {
            "type": "text_delta",
            "content_index": proxy_event["contentIndex"],
            "delta": proxy_event["delta"],
            "partial": partial,
        }

    if event_type == "text_end":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError("Received text_end for non-text content")
        if proxy_event.get("contentSignature") is not None:
            block["textSignature"] = proxy_event["contentSignature"]
        return {
            "type": "text_end",
            "content_index": proxy_event["contentIndex"],
            "content": block["text"],
            "partial": partial,
        }

    if event_type == "thinking_start":
        _set_content(proxy_event["contentIndex"], {"type": "thinking", "thinking": ""})
        return {
            "type": "thinking_start",
            "content_index": proxy_event["contentIndex"],
            "partial": partial,
        }

    if event_type == "thinking_delta":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "thinking":
            raise ValueError("Received thinking_delta for non-thinking content")
        block["thinking"] += proxy_event["delta"]
        return {
            "type": "thinking_delta",
            "content_index": proxy_event["contentIndex"],
            "delta": proxy_event["delta"],
            "partial": partial,
        }

    if event_type == "thinking_end":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "thinking":
            raise ValueError("Received thinking_end for non-thinking content")
        if proxy_event.get("contentSignature") is not None:
            block["thinkingSignature"] = proxy_event["contentSignature"]
        return {
            "type": "thinking_end",
            "content_index": proxy_event["contentIndex"],
            "content": block["thinking"],
            "partial": partial,
        }

    if event_type == "toolcall_start":
        _set_content(
            proxy_event["contentIndex"],
            {
                "type": "toolCall",
                "id": proxy_event["id"],
                "name": proxy_event["toolName"],
                "arguments": {},
                "partialJson": "",
            },
        )
        return {
            "type": "toolcall_start",
            "content_index": proxy_event["contentIndex"],
            "partial": partial,
        }

    if event_type == "toolcall_delta":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            raise ValueError("Received toolcall_delta for non-toolCall content")
        block["partialJson"] += proxy_event["delta"]
        block["arguments"] = partial_json(block["partialJson"]) or {}
        return {
            "type": "toolcall_delta",
            "content_index": proxy_event["contentIndex"],
            "delta": proxy_event["delta"],
            "partial": partial,
        }

    if event_type == "toolcall_end":
        block = content[proxy_event["contentIndex"]]
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            return None
        raw_arguments = block.pop("partialJson", "")
        block["raw_arguments"] = raw_arguments
        return cast(
            AssistantMessageEvent,
            {
                "type": "toolcall_end",
                "content_index": proxy_event["contentIndex"],
                "tool_call": cast(ToolCall, block),
                "partial": partial,
            },
        )

    if event_type == "done":
        done_reason: Any = proxy_event["reason"]
        if done_reason == "toolUse":
            done_reason = "tool_call"
        partial["stop_reason"] = done_reason
        usage = proxy_event.get("usage")
        if isinstance(usage, dict):
            partial["usage"] = cast(Usage, usage)
        return cast(
            AssistantMessageEvent, {"type": "done", "reason": done_reason, "message": partial}
        )

    if event_type == "error":
        error_reason: Literal["aborted", "error"] = (
            "aborted" if proxy_event.get("reason") == "aborted" else "error"
        )
        partial["stop_reason"] = error_reason
        if proxy_event.get("errorMessage") is not None:
            partial["error_message"] = proxy_event["errorMessage"]
        usage = proxy_event.get("usage")
        if isinstance(usage, dict):
            partial["usage"] = cast(Usage, usage)
        return {"type": "error", "reason": error_reason, "error": partial}

    return None


async def _consume_proxy_stream(
    stream: ProxyMessageEventStream,
    model: Model,
    context: Context,
    options: dict[str, Any],
    client: httpx.AsyncClient,
) -> None:
    partial: AssistantMessage = {
        "role": "assistant",
        "stop_reason": "pending",
        "content": [],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": _empty_usage(),
        "timestamp": now_ms(),
    }
    signal = options.get("signal")

    try:
        async with client.stream(
            "POST",
            f"{options['proxyUrl']}/api/stream",
            headers={
                "Authorization": f"Bearer {options['authToken']}",
                "Content-Type": "application/json",
            },
            json={
                "model": _to_jsonable(model),
                "context": _to_jsonable(context),
                "options": build_proxy_request_options(options),
            },
        ) as response:
            if response.status_code != 200:
                error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                try:
                    error_data = response.json()
                    if error_data.get("error"):
                        error_message = f"Proxy error: {error_data['error']}"
                except Exception:
                    pass
                raise RuntimeError(error_message)

            buffer = ""
            async for chunk in response.aiter_text():
                if signal is not None and signal.is_set():
                    raise RuntimeError("Request aborted by user")
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data:
                        continue
                    proxy_event = json.loads(data)
                    event = process_proxy_event(proxy_event, partial)
                    if event is not None:
                        stream.push(event)
        stream.end()
    except BaseException as error:
        reason: Literal["aborted", "error"] = (
            "aborted" if (signal is not None and signal.is_set()) else "error"
        )
        partial["stop_reason"] = reason
        partial["error_message"] = str(error)
        stream.push(
            {
                "type": "error",
                "reason": reason,
                "error": partial,
            }
        )
        stream.end()


def stream_proxy(
    model: Model,
    context: Context,
    options: dict[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
) -> ProxyMessageEventStream:
    """创建代理流：后台任务消费 SSE 事件并重建 partial 消息。"""
    stream = ProxyMessageEventStream()
    owns_client = client is None
    active_client = client or httpx.AsyncClient()

    async def _run() -> None:
        try:
            await _consume_proxy_stream(stream, model, context, options, active_client)
        finally:
            if owns_client:
                await active_client.aclose()

    asyncio.create_task(_run())
    return stream
