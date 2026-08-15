"""OpenAI Codex Responses API（复用 responses.py 解析管线）。

通过 client_factory 注入 AsyncOpenAI，并设置 Codex backend 所需 headers。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

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
from ..utils.uuid import uuidv7
from ..auth.oauth.openai_codex import get_account_id
from ._shared import close_async_client, empty_usage, parse_tool_arguments
from .responses import _build_responses_request_kwargs, _parse_response_usage, responses_stream

_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"


_websocket_sse_fallback_sessions: set[str] = set()
# 回退标记集合的上限：防止 session_id 长期累积无界增长。
_WEBSOCKET_SSE_FALLBACK_MAX_ENTRIES = 256


class _CodexWsConnectError(Exception):
    """WS 连接阶段失败（未产生任何事件），可回退 SSE。"""


class _CodexWsStreamError(Exception):
    """WS 已开始事件流后的传输失败，不回退。"""


def _is_ws_sse_fallback_active(session_id: str | None) -> bool:
    """会话是否已标记 WS→SSE 回退（对齐 TS websocketSseFallbackSessions）。"""
    return bool(session_id) and session_id in _websocket_sse_fallback_sessions


def _record_ws_failure(session_id: str | None) -> None:
    if session_id:
        if len(_websocket_sse_fallback_sessions) >= _WEBSOCKET_SSE_FALLBACK_MAX_ENTRIES:
            _websocket_sse_fallback_sessions.clear()
        _websocket_sse_fallback_sessions.add(session_id)


def _codex_headers(
    api_key: str,
    options: dict[str, Any],
    *,
    compressed: bool = False,
    session_id: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
        "originator": "pi",
        "User-Agent": "pi (python)",
    }
    if compressed:
        headers["content-encoding"] = "zstd"
    # 对齐 TS：从 JWT 中提取 chatgpt-account-id。OAuth 凭据的 token
    # 携带该 claim；显式 option 仍可覆盖。
    account_id = options.get("chatgpt_account_id") or get_account_id(api_key) or ""
    if account_id:
        headers["chatgpt-account-id"] = account_id
    if session_id:
        headers["session-id"] = session_id
        headers["x-client-request-id"] = session_id
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


def _resolve_codex_url(base_url: str) -> str:
    raw = (base_url or _DEFAULT_BASE_URL).rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw
    if raw.endswith("/codex"):
        return f"{raw}/responses"
    return f"{raw}/codex/responses"


def _resolve_codex_websocket_url(base_url: str) -> str:
    endpoint = _resolve_codex_url(base_url)
    if endpoint.startswith("https://"):
        return "wss://" + endpoint[len("https://") :]
    if endpoint.startswith("http://"):
        return "ws://" + endpoint[len("http://") :]
    return endpoint


def _to_event_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_event_obj(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_event_obj(item) for item in value]
    return value


def _codex_websocket_headers(
    api_key: str,
    options: dict[str, Any],
    request_id: str,
) -> dict[str, str]:
    headers = _codex_headers(api_key, options)
    headers.pop("accept", None)
    headers.pop("content-type", None)
    headers.pop("content-encoding", None)
    headers["OpenAI-Beta"] = "responses_websockets=2026-02-06"
    headers["session-id"] = request_id
    headers["x-client-request-id"] = request_id
    return headers


@dataclass(slots=True)
class _CodexWebSocketCacheEntry:
    url: str
    headers: dict[str, str]
    options: dict[str, Any]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ws: Any = None
    last_request_body: dict[str, Any] | None = None
    last_response_id: str | None = None
    last_response_items: list[dict[str, Any]] = field(default_factory=list)


_codex_websocket_cache: dict[str, _CodexWebSocketCacheEntry] = {}
_CODEX_WEBSOCKET_CACHE_MAX_ENTRIES = 64


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _request_bodies_match_except_input(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left = {k: v for k, v in left.items() if k not in ("input", "previous_response_id")}
    right = {k: v for k, v in right.items() if k not in ("input", "previous_response_id")}
    return _json_key(left) == _json_key(right)


def _build_cached_websocket_body(
    entry: _CodexWebSocketCacheEntry,
    body: dict[str, Any],
) -> dict[str, Any]:
    """能安全增量时发送 previous_response_id + 输入 delta（对齐 TS）。"""
    if entry.last_request_body is None or not entry.last_response_id:
        return dict(body)
    if not _request_bodies_match_except_input(entry.last_request_body, body):
        entry.last_request_body = None
        entry.last_response_id = None
        entry.last_response_items = []
        return dict(body)
    baseline = [*entry.last_request_body.get("input", []), *entry.last_response_items]
    current = list(body.get("input", []))
    if len(current) < len(baseline):
        return dict(body)
    prefix = current[: len(baseline)]
    if _json_key(prefix) != _json_key(baseline):
        return dict(body)
    return {
        **body,
        "previous_response_id": entry.last_response_id,
        "input": current[len(baseline) :],
    }


async def _get_or_create_websocket_cache_entry(
    session_id: str,
    url: str,
    headers: dict[str, str],
    options: dict[str, Any],
) -> _CodexWebSocketCacheEntry:
    entry = _codex_websocket_cache.get(session_id)
    if entry is not None and (entry.url != url or _json_key(entry.headers) != _json_key(headers)):
        if entry.ws is not None:
            try:
                await entry.ws.close()
            except Exception:
                pass
        _codex_websocket_cache.pop(session_id, None)
        entry = None
    if entry is None:
        if len(_codex_websocket_cache) >= _CODEX_WEBSOCKET_CACHE_MAX_ENTRIES:
            for old_entry in list(_codex_websocket_cache.values()):
                if old_entry.ws is not None:
                    try:
                        await old_entry.ws.close()
                    except Exception:
                        pass
            _codex_websocket_cache.clear()
        entry = _CodexWebSocketCacheEntry(url=url, headers=headers, options=options)
        _codex_websocket_cache[session_id] = entry
    return entry


class _CodexWebSocketResponses:
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        options: dict[str, Any],
        events: "_CodexWebSocketEvents | None" = None,
    ) -> None:
        self._url = url
        self._headers = headers
        self._options = options
        self._events = events

    async def create(self, **kwargs: Any) -> "_CodexWebSocketEvents":
        if self._events is not None:
            return self._events
        return _CodexWebSocketEvents(
            url=self._url,
            headers=self._headers,
            body=kwargs,
            options=self._options,
        )


class _CodexWebSocketClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        options: dict[str, Any],
        events: "_CodexWebSocketEvents | None" = None,
    ) -> None:
        self.responses = _CodexWebSocketResponses(url, headers, options, events)
        self._events = events

    async def close(self) -> None:
        """关闭底层 WebSocket（responses.py 的 finally 兜底路径）。"""
        if self._events is not None:
            await self._events.aclose()


class _CodexWebSocketEvents:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        options: dict[str, Any],
        cache_entry: _CodexWebSocketCacheEntry | None = None,
        full_body: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._headers = headers
        self._body = body
        self._full_body = full_body or body
        self._options = options
        self._events = False
        self._ws: Any = None
        self._closed = False
        self._cache_entry = cache_entry
        self._keep_alive = False
        self._lock_released = False

    def __aiter__(self) -> "_CodexWebSocketEvents":
        return self

    async def connect(self) -> None:
        """建立 WebSocket 连接并发送 response.create（连接失败抛 _CodexWsConnectError）。"""
        await self._open()

    async def __anext__(self) -> Any:
        while True:
            if self._closed:
                raise StopAsyncIteration
            if not self._events:
                await self._open()
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=float(self._options.get("timeout_ms") or 300000) / 1000.0,
                )
            except asyncio.CancelledError:
                # 消费方取消：必须关闭连接再传播，否则 socket 泄漏。
                await self.aclose()
                raise
            except Exception as exc:
                await self.aclose()
                # 已开始事件流后的传输失败：不回退（对齐 TS websocketStarted → throw）。
                raise _CodexWsStreamError(f"Codex WebSocket stream failed: {exc}") from exc
            text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type in (
                "response.completed",
                "response.done",
                "response.incomplete",
                "response.failed",
            ):
                try:
                    return _to_event_obj(event)
                finally:
                    self._record_terminal(event)
                    await self.aclose()
                    self._closed = True
            return _to_event_obj(event)

    def _record_terminal(self, event: dict[str, Any]) -> None:
        entry = self._cache_entry
        if entry is None:
            return
        if event.get("type") not in ("response.completed", "response.done"):
            # incomplete/failed 不能作为 cached continuation 基线；连接也不复用。
            entry.last_request_body = None
            entry.last_response_id = None
            entry.last_response_items = []
            entry.ws = None
            self._keep_alive = False
            return
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                entry.last_response_id = response_id
            output = response.get("output")
            entry.last_response_items = (
                [item for item in output if isinstance(item, dict)]
                if isinstance(output, list)
                else []
            )
        entry.last_request_body = dict(self._full_body)
        entry.ws = self._ws
        self._keep_alive = True

    def _release_lock(self) -> None:
        entry = self._cache_entry
        if entry is not None and not self._lock_released:
            entry.lock.release()
            self._lock_released = True

    async def _open(self) -> None:
        import websockets

        entry = self._cache_entry
        if entry is not None and entry.ws is not None:
            self._ws = entry.ws
        else:
            timeout_ms = int(self._options.get("websocket_connect_timeout_ms") or 15000)
            try:
                self._ws = await websockets.connect(
                    self._url,
                    additional_headers=self._headers,
                    open_timeout=timeout_ms / 1000.0,
                )
            except Exception as exc:
                self._ws = None
                raise _CodexWsConnectError(f"Codex WebSocket connect failed: {exc}") from exc
            if entry is not None:
                entry.ws = self._ws
        try:
            await self._ws.send(json.dumps({"type": "response.create", **self._body}))
        except Exception as exc:
            # 连接/发送阶段失败（未产生任何事件）：可回退 SSE
            # （对齐 TS websocketStarted=false）。
            try:
                await self._ws.close()
            except Exception:
                pass
            if entry is not None and entry.ws is self._ws:
                entry.ws = None
            self._ws = None
            raise _CodexWsConnectError(f"Codex WebSocket connect failed: {exc}") from exc
        self._events = True

    async def aclose(self) -> None:
        entry = self._cache_entry
        if self._keep_alive:
            # cached connection 保留给同 session 的下一次请求。
            self._release_lock()
            return
        if getattr(self, "_ws", None) is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            if entry is not None and entry.ws is self._ws:
                entry.ws = None
            self._ws = None
        self._release_lock()


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
    transport = opts.get("transport")
    use_websocket = transport in ("websocket", "websocket-cached", "auto")
    endpoint = _resolve_codex_client_base_url(base_url or model.base_url or "")
    request_id = str(opts.get("session_id") or uuidv7())
    session_id = request_id
    if use_websocket and not _is_ws_sse_fallback_active(session_id):
        try:
            return await _websocket_stream(
                model, context, api_key, base_url, endpoint, opts, request_id
            )
        except _CodexWsConnectError:
            # 连接阶段失败：记录会话级回退并转 SSE（对齐 TS
            # recordWebSocketSseFallback → SSE 路径）。
            # 仅记录调用方显式提供的稳定 session_id：每次请求生成的
            # 一次性 uuid 永不复用，记录只会无界增长且回退记忆不生效。
            if opts.get("session_id"):
                _record_ws_failure(session_id)

    codex_headers = _codex_headers(
        api_key,
        opts,
        compressed=True,
        session_id=cast(str | None, opts.get("session_id")),
    )

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


async def _websocket_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str,
    endpoint: str,
    opts: dict[str, Any],
    request_id: str,
) -> AssistantMessageEventStream:
    ws_url = _resolve_codex_websocket_url(base_url or model.base_url or "")
    ws_headers = _codex_websocket_headers(api_key, opts, request_id)
    # 先构造完整请求体并提前建立连接：连接失败在调用点同步抛出
    # _CodexWsConnectError，让 openai_codex_responses_stream 的 except
    # 分支能真正触发 SSE 回退（延迟到后台任务连接则永远到不了调用点）。
    kwargs, _web_search = _build_responses_request_kwargs(
        model,
        context,
        endpoint,
        cast(StreamOptions, opts),
        request_model_id=model.id,
    )
    use_cached_context = opts.get("transport") in ("websocket-cached", "auto") and bool(
        opts.get("session_id")
    )
    cache_entry = None
    if use_cached_context:
        cache_entry = await _get_or_create_websocket_cache_entry(
            request_id, ws_url, ws_headers, opts
        )
        await cache_entry.lock.acquire()
    try:
        request_body = (
            _build_cached_websocket_body(cache_entry, kwargs) if cache_entry is not None else kwargs
        )
        events = _CodexWebSocketEvents(
            url=ws_url,
            headers=ws_headers,
            body=request_body,
            options=opts,
            cache_entry=cache_entry,
            full_body=kwargs,
        )
        await events.connect()
    except BaseException:
        if cache_entry is not None:
            cache_entry.lock.release()
        raise

    def _ws_factory(
        _api_key: str,
        _base_url: str,
        *,
        timeout: float,
        max_retries: int,
        headers: dict[str, str] | None,
    ) -> _CodexWebSocketClient:
        return _CodexWebSocketClient(ws_url, ws_headers, opts, events)

    return await responses_stream(
        model,
        context,
        api_key,
        endpoint,
        cast(StreamOptions, opts),
        client_factory=cast(Callable[..., Any], _ws_factory),
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
    try:
        response = await client.responses.retrieve(handle["id"])
    finally:
        await close_async_client(client)
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
    raw_usage = getattr(response, "usage", None)
    # 与 responses_stream 共用 usage 解析：input_tokens 已含缓存 token，
    # 统一扣减 cache_read/cache_write，避免双重计费。
    usage = _parse_response_usage(response, model) if raw_usage is not None else empty_usage()
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
    try:
        await client.responses.cancel(handle["id"])
    finally:
        await close_async_client(client)


__all__ = [
    "openai_codex_responses_stream",
    "codex_stream",
    "codex_stream_simple",
    "codex_fetch_deferred",
    "codex_cancel_deferred",
]
