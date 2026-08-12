"""AWS Bedrock ConverseStream（bearer token + EventStream 解析）。

实现核心的 text / reasoning / tool use 流，认证使用 bearer token，
不引入 AWS SDK 依赖。
"""

from __future__ import annotations

import asyncio
import configparser
import hashlib
import hmac
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

from typing import Any, AsyncIterator, cast

import httpx

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
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.cost import calculate_cost
from ..utils.provider_env import get_provider_env_value
from ._shared import build_error_message, empty_usage, parse_tool_arguments
from .simple_options import clamp_max_tokens_to_context

_AsyncClient = httpx.AsyncClient
_DEFAULT_REGION = "us-east-1"


def _resolve_bedrock_credentials(
    options: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """解析 AWS 凭证：显式/env > shared credentials file（默认 profile）。"""
    env = options.get("env")
    access_key = cast(str | None, options.get("aws_access_key_id")) or get_provider_env_value(
        "AWS_ACCESS_KEY_ID", env
    )
    secret_key = cast(str | None, options.get("aws_secret_access_key")) or (
        get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    )
    session_token = cast(str | None, options.get("aws_session_token")) or (
        get_provider_env_value("AWS_SESSION_TOKEN", env)
    )
    if access_key and secret_key:
        return access_key, secret_key, session_token

    profile = (
        options.get("aws_profile")
        or get_provider_env_value("AWS_PROFILE", env)
        or get_provider_env_value("AWS_DEFAULT_PROFILE", env)
        or "default"
    )
    raw_path = (
        options.get("aws_shared_credentials_file")
        or get_provider_env_value("AWS_SHARED_CREDENTIALS_FILE", env)
        or Path.home() / ".aws" / "credentials"
    )
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        return access_key, secret_key, session_token
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return access_key, secret_key, session_token
    if not parser.has_section(profile):
        return access_key, secret_key, session_token
    section = parser[profile]
    return (
        section.get("aws_access_key_id"),
        section.get("aws_secret_access_key"),
        section.get("aws_session_token") or session_token,
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aws_sigv4_headers(
    *,
    method: str,
    url: str,
    payload: bytes,
    region: str,
    access_key: str,
    secret_key: str,
    session_token: str | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    parsed = urlparse(url)
    timestamp = now or datetime.now(timezone.utc)
    amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp.strftime("%Y%m%d")
    host = parsed.netloc
    path = quote(parsed.path, safe="/")
    payload_hash = _sha256_hex(payload)
    headers = {
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "content-type": "application/json",
    }
    if session_token:
        headers["x-amz-security-token"] = session_token
    canonical_headers = "".join(f"{k}:{v}\n" for k, v in sorted(headers.items()))
    signed_headers = ";".join(sorted(headers))
    canonical_request = "\n".join(
        [method, path, "", canonical_headers, signed_headers, payload_hash]
    )
    credential_scope = f"{date_stamp}/{region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request.encode("utf-8")),
        ]
    )

    def _hmac(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()

    date_key = _hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _hmac(date_key, region)
    service_key = _hmac(region_key, "bedrock")
    signing_key = _hmac(service_key, "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "content-type": "application/json",
    }


def _read_header_value(data: bytes, offset: int) -> tuple[int, str, str]:
    name_length = data[offset]
    offset += 1
    name = data[offset : offset + name_length].decode("utf-8")
    offset += name_length
    value_type = data[offset]
    offset += 1
    value_length = struct.unpack(">H", data[offset : offset + 2])[0]
    offset += 2
    value = data[offset : offset + value_length]
    offset += value_length
    if value_type == 7:  # string
        return offset, name, value.decode("utf-8")
    return offset, name, value.decode("utf-8", errors="replace")


def parse_eventstream_messages(
    buffer: bytes,
) -> tuple[list[dict[str, Any]], bytes]:
    """解析 AWS EventStream 二进制消息，返回 (payload dicts, remainder)。"""
    messages: list[dict[str, Any]] = []
    offset = 0
    while offset + 16 <= len(buffer):
        frame_start = offset
        total_length, headers_length = struct.unpack(">II", buffer[offset : offset + 8])
        if total_length < 16 or total_length > len(buffer) - offset:
            break
        offset += 12  # skip prelude + prelude crc
        headers_end = offset + headers_length
        headers: dict[str, str] = {}
        pos = offset
        while pos < headers_end:
            pos, name, value = _read_header_value(buffer, pos)
            headers[name] = value
        offset = headers_end
        payload_end = frame_start + total_length - 4  # crc
        payload = buffer[offset:payload_end]
        offset = payload_end + 4
        event_type = headers.get(":event-type", "")
        try:
            payload_obj: Any = json.loads(payload.decode("utf-8"))
        except Exception:
            payload_obj = {}
        if isinstance(payload_obj, dict):
            payload_obj["_event_type"] = event_type
            messages.append(payload_obj)
    return messages, buffer[offset:]


async def read_bedrock_events(
    bytes_iter: AsyncIterator[bytes],
) -> AsyncIterator[dict[str, Any]]:
    buffer = b""
    async for chunk in bytes_iter:
        buffer += chunk
        messages, buffer = parse_eventstream_messages(buffer)
        for message in messages:
            yield message


def _to_bedrock_messages(context: Context) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        role = msg["role"]
        if role == "user":
            content = cast(Any, msg["content"])
            if isinstance(content, str):
                user_blocks: list[dict[str, Any]] = [{"text": content}]
            else:
                user_blocks = []
                for block in content:
                    if block["type"] == "text":
                        user_blocks.append({"text": block["text"]})
                    elif block["type"] == "image" and block.get("data"):
                        fmt = {
                            "image/png": "png",
                            "image/jpeg": "jpeg",
                            "image/gif": "gif",
                            "image/webp": "webp",
                        }.get(block.get("mime_type") or "image/png", "png")
                        user_blocks.append(
                            {
                                "image": {
                                    "format": fmt,
                                    "source": {"bytes": block["data"]},
                                }
                            }
                        )
            messages.append({"role": "user", "content": user_blocks})
        elif role == "assistant":
            assistant_msg = cast(Any, msg)
            assistant_blocks: list[dict[str, Any]] = []
            for block in assistant_msg.get("content") or []:
                if block["type"] == "text":
                    assistant_blocks.append({"text": block["text"]})
                elif block["type"] == "toolCall":
                    assistant_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": block["id"],
                                "name": block["name"],
                                "input": block.get("arguments") or {},
                            }
                        }
                    )
            if assistant_blocks:
                messages.append({"role": "assistant", "content": assistant_blocks})
        elif role == "toolResult":
            tool_msg = cast(Any, msg)
            text = "".join(
                block.get("text") or "" for block in tool_msg["content"] if block["type"] == "text"
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_msg["tool_call_id"],
                                "content": [{"text": text}],
                                "status": "error" if tool_msg.get("is_error") else "success",
                            }
                        }
                    ],
                }
            )
    return messages


def _to_bedrock_tools(context: Context) -> list[dict[str, Any]] | None:
    if not context.tools:
        return None
    return [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.input_schema},
            }
        }
        for tool in context.tools
    ]


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason == "end_turn":
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_call"
    return "stop"


def _update_usage(model: Model, usage: Usage, metadata: dict[str, Any]) -> None:
    raw = metadata.get("usage") or {}
    usage["input"] = int(raw.get("inputTokens") or 0)
    usage["output"] = int(raw.get("outputTokens") or 0)
    usage["cache_read"] = int(raw.get("cacheReadInputTokens") or 0)
    usage["cache_write"] = int(raw.get("cacheWriteInputTokens") or 0)
    usage["total_tokens"] = int(raw.get("totalTokens") or 0) or (usage["input"] + usage["output"])
    calculate_cost(model, usage)


def bedrock_converse_stream(
    model: Model,
    context: Context,
    token: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    opts = options or {}
    messages = _to_bedrock_messages(context)
    tools = _to_bedrock_tools(context)
    requested = opts.get("max_tokens")
    max_tokens = requested if requested is not None else model.max_tokens

    payload: dict[str, Any] = {
        "modelId": model.id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": clamp_max_tokens_to_context(model, context, max_tokens)},
    }
    temperature = opts.get("temperature")
    if temperature is not None:
        payload["inferenceConfig"]["temperature"] = temperature
    if context.system_prompt:
        payload["system"] = [{"text": context.system_prompt}]
    if tools:
        payload["toolConfig"] = {"tools": tools}

    async def _run() -> None:
        content_blocks: list[ContentBlock] = []
        current_index: int | None = None
        current_kind: str | None = None
        current_tool_id = ""
        current_tool_name = ""
        current_raw_args = ""
        usage: Usage = empty_usage()
        stop_reason: StopReason = "stop"

        def _partial() -> AssistantMessage:
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
            nonlocal current_kind, current_index
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
                        content=thinking_block.get("thinking") or "",
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

        try:
            region = (
                opts.get("region")
                or get_provider_env_value("AWS_REGION", opts.get("env"))
                or _DEFAULT_REGION
            )
            endpoint = (
                base_url
                or f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.id}/converse-stream"
            )
            headers: dict[str, str] = {
                "accept": "application/vnd.amazon.eventstream",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                access_key, secret_key, session_token = _resolve_bedrock_credentials(
                    cast(dict[str, Any], opts)
                )
                if not access_key or not secret_key:
                    raise RuntimeError(
                        f"No AWS credentials or bearer token for provider: {model.provider}. "
                        "Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, configure "
                        "~/.aws/credentials, or pass a bearer token."
                    )
                sig_headers = _aws_sigv4_headers(
                    method="POST",
                    url=endpoint,
                    payload=json.dumps(payload).encode("utf-8"),
                    region=str(region),
                    access_key=access_key,
                    secret_key=secret_key,
                    session_token=session_token,
                )
                headers.update(sig_headers)
            for name, value in (opts.get("headers") or {}).items():
                if value is not None:
                    headers[name] = value
            timeout_ms = opts.get("timeout_ms") or 120000
            async with _AsyncClient(timeout=timeout_ms / 1000) as client:
                async with client.stream(
                    "POST", endpoint, headers=headers, json=payload
                ) as response:
                    if not response.is_success:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        raise RuntimeError(f"Bedrock API error ({response.status_code}): {body}")
                    stream.push(StartEvent(type="start", partial=_partial()))
                    async for event in read_bedrock_events(response.aiter_bytes()):
                        etype = event.pop("_event_type", "")
                        if etype == "messageStart":
                            continue
                        if etype == "contentBlockStart":
                            start = event.get("start") or {}
                            if start.get("toolUse"):
                                _end_current_block()
                                current_tool_id = start["toolUse"].get("toolUseId") or ""
                                current_tool_name = start["toolUse"].get("name") or ""
                                current_raw_args = ""
                                content_blocks.append(
                                    ToolCall(
                                        type="toolCall",
                                        id=current_tool_id,
                                        name=current_tool_name,
                                        raw_arguments="",
                                        arguments=None,
                                    )
                                )
                                current_index = len(content_blocks) - 1
                                current_kind = "toolCall"
                                stream.push(
                                    ToolCallStartEvent(
                                        type="toolcall_start",
                                        content_index=current_index,
                                        partial=_partial(),
                                    )
                                )
                        elif etype == "contentBlockDelta":
                            delta = event.get("delta") or {}
                            if "text" in delta:
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
                                index = cast(int, current_index)
                                text_block = cast(TextContent, content_blocks[index])
                                text = delta["text"]
                                text_block["text"] += text
                                stream.push(
                                    TextDeltaEvent(
                                        type="text_delta",
                                        content_index=index,
                                        delta=text,
                                        partial=_partial(),
                                    )
                                )
                            elif delta.get("toolUse"):
                                index = cast(int, current_index)
                                chunk = delta["toolUse"].get("input") or ""
                                current_raw_args += chunk
                                stream.push(
                                    ToolCallDeltaEvent(
                                        type="toolcall_delta",
                                        content_index=index,
                                        delta=chunk,
                                        partial=_partial(),
                                    )
                                )
                            elif delta.get("reasoningContent"):
                                if current_kind != "thinking":
                                    _end_current_block()
                                    current_kind = "thinking"
                                    content_blocks.append(
                                        ThinkingContent(type="thinking", thinking="")
                                    )
                                    current_index = len(content_blocks) - 1
                                    stream.push(
                                        ThinkingStartEvent(
                                            type="thinking_start",
                                            content_index=current_index,
                                            partial=_partial(),
                                        )
                                    )
                                index = cast(int, current_index)
                                thinking_block = cast(ThinkingContent, content_blocks[index])
                                reasoning = delta["reasoningContent"].get("reasoningText") or {}
                                text = reasoning.get("text") or ""
                                if text:
                                    thinking_block["thinking"] += text
                                    stream.push(
                                        ThinkingDeltaEvent(
                                            type="thinking_delta",
                                            content_index=index,
                                            delta=text,
                                            partial=_partial(),
                                        )
                                    )
                        elif etype == "contentBlockStop":
                            _end_current_block()
                        elif etype == "messageStop":
                            stop_reason = _map_stop_reason(event.get("stopReason"))
                        elif etype == "metadata":
                            _update_usage(model, usage, event)
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
            assert stop_reason in ("stop", "length", "tool_call")
            stream.push({"type": "done", "reason": stop_reason, "message": msg})
        except asyncio.CancelledError:
            stream.error(asyncio.CancelledError())
            raise
        except Exception as exc:
            err_msg = build_error_message(model, exc)
            stream.push({"type": "error", "reason": "error", "error": err_msg})

    asyncio.create_task(_run())
    return stream


def bedrock_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    return bedrock_converse_stream(
        model,
        context,
        opts.get("api_key") or "",
        opts.get("base_url") or model.base_url or "",
        options,
    )


def bedrock_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return bedrock_stream(model, context, options)


__all__ = [
    "bedrock_converse_stream",
    "bedrock_stream",
    "bedrock_stream_simple",
    "parse_eventstream_messages",
]
