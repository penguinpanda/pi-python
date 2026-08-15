"""Mistral Conversations API（复用 OpenAI 兼容 completions 模式）。

对齐 TS packages/ai/src/api/mistral-conversations.ts 的调用面；
Mistral 提供 OpenAI 兼容的 /v1/chat/completions 端点，因此复用
chat_completions_stream，并补充 x-affinity 前缀缓存头。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from ..types import (
    AssistantMessage,
    Context,
    Model,
    StreamOptions,
)
from ..utils._event_stream import AssistantMessageEventStream
from .completions import chat_completions_stream
from .transform_messages import short_hash

_MISTRAL_TOOL_CALL_ID_LENGTH = 9


def _derive_mistral_tool_call_id(id_: str, attempt: int) -> str:
    """按 TS deriveMistralToolCallId 生成 9 位字母数字 ID。"""
    normalized = re.sub(r"[^a-zA-Z0-9]", "", id_)
    if attempt == 0 and len(normalized) == _MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or id_
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return re.sub(r"[^a-zA-Z0-9]", "", short_hash(seed))[:_MISTRAL_TOOL_CALL_ID_LENGTH]


def create_mistral_tool_call_id_normalizer() -> Callable[[str, Model, AssistantMessage], str]:
    """Mistral 专用 tool call ID 规范化（每次请求独立映射）。"""
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(id_: str, model: Model, source: AssistantMessage) -> str:
        existing = id_map.get(id_)
        if existing is not None:
            return existing
        attempt = 0
        while True:
            candidate = _derive_mistral_tool_call_id(id_, attempt)
            owner = reverse_map.get(candidate)
            if owner is None or owner == id_:
                id_map[id_] = candidate
                reverse_map[candidate] = id_
                return candidate
            attempt += 1

    return normalize


async def mistral_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    api_key = opts.get("api_key") or ""
    base_url = opts.get("base_url") or model.base_url or "https://api.mistral.ai/v1"
    headers = dict(opts.get("headers") or {})
    session_id = opts.get("session_id")
    cache_retention = opts.get("cache_retention")
    if session_id and cache_retention != "none" and "x-affinity" not in headers:
        headers["x-affinity"] = session_id
    request_options = dict(opts)
    if session_id and cache_retention != "none":
        extra_body = dict(cast(dict[str, Any], request_options.get("extra_body")) or {})
        extra_body.setdefault("prompt_cache_key", session_id)
        request_options["extra_body"] = extra_body
    if headers:
        merged = dict(cast(dict[str, str | None] | None, request_options.get("headers")) or {})
        merged.update(headers)
        request_options["headers"] = merged
    return await chat_completions_stream(
        model,
        context,
        api_key,
        base_url,
        cast(StreamOptions, request_options),
        tool_call_id_normalizer=create_mistral_tool_call_id_normalizer(),
    )


async def mistral_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return await mistral_stream(model, context, options)


__all__ = ["mistral_stream", "mistral_stream_simple"]
