"""pi-messages API 实现（对齐 TS api/pi-messages.ts）。

pi 自有线协议（Radius 网关等后端实现）：单次 POST
`{baseUrl}/messages`，请求体为 {model, context, options}，
响应为 SSE 的 assistant-message 事件流 + 终态 done/error 事件。
"""

import asyncio
import json

from typing import Any, AsyncIterator

import httpx

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Model,
    ProviderStreams,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    ToolCall,
    Usage,
    now_ms,
)
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.diagnostics import append_assistant_message_diagnostic
from ..utils.json_parse import parse_streaming_json

_AsyncClient = httpx.AsyncClient


class PiMessagesResponseError(Exception):
    """非 2xx 响应错误（携带错误码与诊断详情）。"""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        diagnostic_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic_details = diagnostic_details or {}


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )


def _map_done_reason(reason: str | None) -> StopReason:
    return "tool_call" if reason == "toolUse" else (reason or "stop")


def _context_payload(context: Context) -> dict[str, Any]:
    """Context → 线协议 JSON 字典。"""
    return {
        "messages": context.messages,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in context.tools
        ],
        "system_prompt": context.system_prompt,
        "metadata": context.metadata,
    }


def _append_rewrite_diagnostic(
    message: AssistantMessage, rewrite: dict[str, Any] | None
) -> None:
    if not rewrite:
        return
    append_assistant_message_diagnostic(
        message,
        {
            "type": "pi_messages_rewrite",
            "timestamp": now_ms(),
            "details": dict(rewrite),
        },
    )


def _set_content_block(
    partial: AssistantMessage,
    content_index: int,
    block: dict[str, Any],
) -> dict[str, Any]:
    """写入 content[content_index]；JS 数组可越界赋值，Python 需先扩展。"""
    content = partial["content"]
    while len(content) <= content_index:
        content.append(None)
    content[content_index] = block
    return block


def _create_event_converter(model: Model):
    """构造 wire 事件 → SDK AssistantMessageEvent 的转换器。"""
    partial: AssistantMessage = {
        "role": "assistant",
        "content": [],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": _empty_usage(),
        "stop_reason": "pending",
        "timestamp": now_ms(),
    }
    tool_json: dict[int, str] = {}

    def convert(event: dict[str, Any]) -> AssistantMessageEvent:
        etype = event.get("type")
        if etype == "done":
            partial["stop_reason"] = _map_done_reason(event.get("reason"))
            if isinstance(event.get("usage"), dict):
                partial["usage"] = event["usage"]
            if event.get("responseId"):
                partial["response_id"] = event["responseId"]
            _append_rewrite_diagnostic(partial, event.get("rewrite"))
            return {"type": "done", "reason": partial["stop_reason"], "message": partial}
        if etype == "error":
            reason = "aborted" if event.get("reason") == "aborted" else "error"
            partial["stop_reason"] = reason
            if isinstance(event.get("usage"), dict):
                partial["usage"] = event["usage"]
            if event.get("errorMessage"):
                partial["error_message"] = event["errorMessage"]
            if event.get("responseId"):
                partial["response_id"] = event["responseId"]
            _append_rewrite_diagnostic(partial, event.get("rewrite"))
            return {"type": "error", "reason": reason, "error": partial}

        content_index = event.get("contentIndex", 0)
        if etype == "start":
            return {"type": "start", "partial": partial}
        if etype == "text_start":
            _set_content_block(partial, content_index, {"type": "text", "text": ""})
            return {"type": "text_start", "content_index": content_index, "partial": partial}
        if etype == "text_delta":
            block = partial["content"][content_index]
            block["text"] += event.get("delta", "")
            return {"type": "text_delta", "content_index": content_index, "delta": event.get("delta", ""), "partial": partial}
        if etype == "text_end":
            block = partial["content"][content_index]
            block["text"] = event.get("content", "")
            if event.get("contentSignature"):
                block["text_signature"] = event["contentSignature"]
            return {"type": "text_end", "content_index": content_index, "content": block["text"], "partial": partial}
        if etype == "thinking_start":
            _set_content_block(partial, content_index, {"type": "thinking", "thinking": ""})
            return {"type": "thinking_start", "content_index": content_index, "partial": partial}
        if etype == "thinking_delta":
            block = partial["content"][content_index]
            block["thinking"] += event.get("delta", "")
            return {"type": "thinking_delta", "content_index": content_index, "delta": event.get("delta", ""), "partial": partial}
        if etype == "thinking_end":
            block = partial["content"][content_index]
            block["thinking"] = event.get("content", "")
            if event.get("contentSignature"):
                block["thinking_signature"] = event["contentSignature"]
            if event.get("redacted"):
                block["redacted"] = True
            return {"type": "thinking_end", "content_index": content_index, "content": block["thinking"], "partial": partial}
        if etype == "toolcall_start":
            _set_content_block(
                partial,
                content_index,
                {
                    "type": "toolCall",
                    "id": event.get("id", ""),
                    "name": event.get("toolName", ""),
                    "arguments": None,
                    "raw_arguments": "",
                },
            )
            tool_json[content_index] = ""
            return {"type": "toolcall_start", "content_index": content_index, "partial": partial}
        if etype == "toolcall_delta":
            accumulated = tool_json.get(content_index, "") + event.get("delta", "")
            tool_json[content_index] = accumulated
            block = partial["content"][content_index]
            block["arguments"] = parse_streaming_json(accumulated)
            return {"type": "toolcall_delta", "content_index": content_index, "delta": event.get("delta", ""), "partial": partial}
        if etype == "toolcall_end":
            wire_tool = event.get("toolCall") or {}
            block = partial["content"][content_index]
            block["id"] = wire_tool.get("id", block.get("id", ""))
            block["name"] = wire_tool.get("name", block.get("name", ""))
            block["raw_arguments"] = wire_tool.get("raw_arguments") or tool_json.get(content_index, "")
            block["arguments"] = wire_tool.get("arguments") or parse_streaming_json(block["raw_arguments"])
            tool_json.pop(content_index, None)
            return {
                "type": "toolcall_end",
                "content_index": content_index,
                "tool_call": block,
                "partial": partial,
            }
        raise ValueError(f"Unknown pi-messages event type: {etype}")

    return convert


def _parse_pi_message_event(raw: str) -> dict[str, Any] | None:
    data = None
    for line in raw.split("\n"):
        if line.startswith("data:"):
            data = line[5:].strip()
            break
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


async def read_pi_messages_events(
    bytes_iter: AsyncIterator[bytes],
) -> AsyncIterator[dict[str, Any]]:
    """把 SSE 字节流切成 data: 事件。"""
    buffer = ""
    async for chunk in bytes_iter:
        buffer += chunk.decode("utf-8", errors="replace")
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            event = _parse_pi_message_event(raw)
            if event:
                yield event
    if buffer.strip():
        event = _parse_pi_message_event(buffer)
        if event:
            yield event


def _create_response_error(
    model: Model,
    response: httpx.Response,
    body: str,
) -> PiMessagesResponseError:
    try:
        parsed = json.loads(body)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if not isinstance(error, dict):
            error = None
    except Exception:
        error = None
    message = error.get("message") if isinstance(error, dict) and isinstance(error.get("message"), str) else None
    code = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
    suffix = message or body
    code_suffix = f" ({code})" if code else ""
    return PiMessagesResponseError(
        f"{response.status_code} {response.reason_phrase}: {suffix}{code_suffix}",
        code,
        {
            "provider": model.provider,
            "model": model.id,
            "status": response.status_code,
            "error": error,
            "body": body if error is None else None,
            "timestamp_ms": now_ms(),
        },
    )


def _create_error_message(
    model: Model,
    error: BaseException,
    aborted: bool,
) -> AssistantMessage:
    reason = "aborted" if aborted else "error"
    message: AssistantMessage = {
        "role": "assistant",
        "content": [],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": _empty_usage(),
        "stop_reason": reason,
        "error_message": str(error),
        "timestamp": now_ms(),
    }
    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            message,
            {
                "type": "pi_messages_response_failure",
                "timestamp": now_ms(),
                "error": {"name": type(error).__name__, "message": str(error)},
                "details": error.diagnostic_details,
            },
        )
    return message


def _resolve_cache_retention(cache_retention: str | None, env: dict | None) -> str | None:
    if cache_retention:
        return cache_retention
    from ..utils.provider_env import get_provider_env_value

    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else None


def _options_payload(opts: StreamOptions) -> dict[str, Any]:
    return {
        "temperature": opts.get("temperature"),
        "maxTokens": opts.get("max_tokens"),
        "reasoning": opts.get("reasoning"),
        "cacheRetention": _resolve_cache_retention(
            opts.get("cache_retention"), opts.get("env")
        ),
        "sessionId": opts.get("session_id"),
        "toolChoice": opts.get("tool_choice"),
    }


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """POST {baseUrl}/messages 并消费 SSE（同步返回 EventStream）。"""
    outer = AssistantMessageEventStream()
    opts = options or {}
    convert_event = _create_event_converter(model)

    async def _run() -> None:
        try:
            api_key = opts.get("api_key")
            if not api_key:
                raise RuntimeError(f'No API key provided for provider "{model.provider}"')

            base_url = (model.base_url or "").rstrip("/")
            url = f"{base_url}/messages"
            if opts.get("debug"):
                url = f"{url}?debug=1"

            payload: dict[str, Any] = {
                "model": model.id,
                "context": _context_payload(context),
                "options": _options_payload(opts),
            }
            on_payload = opts.get("on_payload")
            if on_payload is not None:
                next_payload = on_payload(payload, model)
                if asyncio.iscoroutine(next_payload):
                    next_payload = await next_payload
                if next_payload is not None:
                    payload = next_payload

            headers: dict[str, str] = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            }
            request_headers = opts.get("headers") or {}
            for name, value in request_headers.items():
                if value is not None:
                    headers[name] = value

            timeout_ms = opts.get("timeout_ms") or 120000
            async with _AsyncClient(timeout=timeout_ms / 1000) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    on_response = opts.get("on_response")
                    if on_response is not None:
                        event = {
                            "status": response.status_code,
                            "headers": dict(response.headers),
                        }
                        result = on_response(event, model)
                        if asyncio.iscoroutine(result):
                            await result
                    if not response.is_success:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        raise _create_response_error(model, response, body)

                    async for pi_event in read_pi_messages_events(
                        response.aiter_bytes()
                    ):
                        event = convert_event(pi_event)
                        outer.push(event)
                        if event["type"] in ("done", "error"):
                            return
            raise RuntimeError(
                f"{model.provider} stream ended without a terminal event"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            aborted = opts.get("signal") is not None and opts["signal"].is_set()
            outer.push(
                {
                    "type": "error",
                    "reason": "aborted" if aborted else "error",
                    "error": _create_error_message(model, exc, aborted),
                }
            )
            outer.end(_create_error_message(model, exc, aborted))

    task = asyncio.create_task(_run())
    _run_tasks.add(task)
    task.add_done_callback(_run_tasks.discard)
    return outer


_run_tasks: set[asyncio.Task] = set()


def streamSimple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """stream 的简化入口：透传 reasoning / toolChoice / debug。"""
    extra = options or {}
    merged: StreamOptions = dict(extra)
    merged["reasoning"] = extra.get("reasoning")
    merged["tool_choice"] = extra.get("tool_choice")
    merged["debug"] = extra.get("debug")
    return stream(model, context, merged)


__all__ = [
    "stream",
    "streamSimple",
    "PiMessagesResponseError",
    "read_pi_messages_events",
]
