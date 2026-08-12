"""OpenAI Codex Responses API（复用 responses.py 解析管线）。

通过 client_factory 注入 AsyncOpenAI，并设置 Codex backend 所需 headers。
"""

from __future__ import annotations

import json
import httpx

from typing import Any, Callable, cast

from openai import AsyncOpenAI

from ..types import (
    AssistantMessage,
    Context,
    DeferredHandle,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    now_ms,
)
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.cost import calculate_cost
from ._shared import empty_usage, parse_tool_arguments
from .responses import responses_stream

_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"


def _codex_headers(
    api_key: str,
    options: dict[str, Any],
    *,
    compressed: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
        "originator": "pi",
    }
    if compressed:
        headers["content-encoding"] = "zstd"
    account_id = options.get("chatgpt_account_id") or ""
    if account_id:
        headers["chatgpt-account-id"] = account_id
    for name, value in (options.get("headers") or {}).items():
        if value is not None:
            headers[name] = value
    return headers


def _resolve_codex_client_base_url(base_url: str) -> str:
    """Codex SDK base：让 AsyncOpenAI 追加 /responses 后落在 /codex/responses。"""
    raw = (base_url or _DEFAULT_BASE_URL).rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw[: -len("/responses")]
    if raw.endswith("/codex"):
        return raw
    return f"{raw}/codex"


def _zstd_compress(data: bytes) -> bytes | None:
    try:
        import zstandard
    except ImportError:
        return None
    try:
        return zstandard.ZstdCompressor(level=3).compress(data)
    except Exception:
        return None


class _CodexZstdTransport(httpx.AsyncBaseTransport):
    """在发送前把标记为 content-encoding: zstd 的请求体压缩。"""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("content-encoding") == "zstd" and request.content:
            compressed = _zstd_compress(request.content)
            if compressed is not None:
                new_request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    content=compressed,
                    extensions=request.extensions,
                )
                new_request.headers["content-length"] = str(len(compressed))
                return await self._inner.handle_async_request(new_request)
        return await self._inner.handle_async_request(request)


async def openai_codex_responses_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = dict(options or {})
    codex_headers = _codex_headers(api_key, opts, compressed=True)
    endpoint = _resolve_codex_client_base_url(base_url or model.base_url or "")

    def _factory(
        _api_key: str,
        _base_url: str,
        *,
        timeout: float,
        max_retries: int,
        headers: dict[str, str] | None,
    ) -> AsyncOpenAI:
        inner = httpx.AsyncHTTPTransport()
        http_client = httpx.AsyncClient(
            transport=_CodexZstdTransport(inner),
            timeout=timeout,
        )
        return AsyncOpenAI(
            api_key=_api_key,
            base_url=endpoint,
            max_retries=max_retries,
            default_headers=codex_headers,
            http_client=http_client,
        )

    return await responses_stream(
        model,
        context,
        api_key,
        endpoint,
        options,
        client_factory=cast(Callable[..., Any], _factory),
        request_model_id=model.id,
    )


async def codex_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    return await openai_codex_responses_stream(
        model,
        context,
        opts.get("api_key") or "",
        opts.get("base_url") or model.base_url or "",
        options,
    )


async def codex_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return await codex_stream(model, context, options)


def _codex_client(api_key: str, endpoint: str, headers: dict[str, str]) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint,
        default_headers=headers,
    )


async def codex_fetch_deferred(
    model: Model,
    handle: DeferredHandle,
    options: dict[str, Any] | None = None,
) -> AssistantMessage:
    opts = dict(options or {})
    api_key = opts.get("api_key") or ""
    headers = _codex_headers(api_key, opts)
    endpoint = (opts.get("base_url") or model.base_url or _DEFAULT_BASE_URL).rstrip("/")
    client = _codex_client(api_key, endpoint, headers)
    response = await client.responses.retrieve(handle["id"])
    content: list[Any] = []
    for item in response.output or []:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "output_text":
                    content.append(TextContent(type="text", text=getattr(block, "text", "")))
        elif getattr(item, "type", None) == "function_call":
            raw = json.dumps(getattr(item, "arguments", None) or {})
            content.append(
                ToolCall(
                    type="toolCall",
                    id=getattr(item, "id", "") or "",
                    name=getattr(item, "name", "") or "",
                    raw_arguments=raw,
                    arguments=parse_tool_arguments(raw),
                )
            )
    usage = empty_usage()
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        usage["input"] = int(getattr(raw_usage, "input_tokens", 0) or 0)
        usage["output"] = int(getattr(raw_usage, "output_tokens", 0) or 0)
        usage["total_tokens"] = usage["input"] + usage["output"]
        calculate_cost(model, usage)
    status = getattr(response, "status", "")
    return AssistantMessage(
        role="assistant",
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=usage,
        stop_reason="length" if status == "incomplete" else "stop",
        error_message=None,
        timestamp=now_ms(),
        response_id=getattr(response, "id", "") or "",
    )


async def codex_cancel_deferred(
    model: Model,
    handle: DeferredHandle,
    options: dict[str, Any] | None = None,
) -> None:
    opts = dict(options or {})
    api_key = opts.get("api_key") or ""
    headers = _codex_headers(api_key, opts)
    endpoint = (opts.get("base_url") or model.base_url or _DEFAULT_BASE_URL).rstrip("/")
    client = _codex_client(api_key, endpoint, headers)
    await client.responses.cancel(handle["id"])


__all__ = [
    "openai_codex_responses_stream",
    "codex_stream",
    "codex_stream_simple",
    "codex_fetch_deferred",
    "codex_cancel_deferred",
]
