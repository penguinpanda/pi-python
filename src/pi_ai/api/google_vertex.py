"""Google Vertex AI（复用 Gemini REST streamGenerateContent + OAuth token）。"""

from __future__ import annotations

from typing import Any, cast

from ..types import (
    Context,
    Model,
    StreamOptions,
)
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from .google_generative_ai import google_generative_ai_stream

_DEFAULT_LOCATION = "us-central1"


def _resolve_vertex_options(
    model: Model,
    api_key: str,
    base_url: str,
    options: dict[str, Any],
) -> tuple[str, str]:
    token = api_key or get_provider_env_value(
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        options.get("env"),
    )
    if not token:
        raise RuntimeError(f"No access token for provider: {model.provider}")
    project = (
        options.get("project")
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.get("env"))
        or get_provider_env_value("GCP_PROJECT", options.get("env"))
    )
    if not project:
        raise RuntimeError("Google Vertex requires a project")
    location = (
        options.get("location")
        or get_provider_env_value("GOOGLE_CLOUD_LOCATION", options.get("env"))
        or _DEFAULT_LOCATION
    )
    endpoint = (
        base_url
        or f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/publishers/google"
    )
    return token, endpoint


async def google_vertex_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = dict(options or {})
    token, endpoint = _resolve_vertex_options(model, api_key, base_url, opts)
    headers = dict(cast(dict[str, str | None] | None, opts.get("headers")) or {})
    headers["Authorization"] = f"Bearer {token}"
    merged = dict(opts)
    merged["headers"] = headers
    return google_generative_ai_stream(
        model,
        context,
        "",
        endpoint,
        cast(StreamOptions, merged),
    )


async def vertex_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    return await google_vertex_stream(
        model,
        context,
        opts.get("api_key") or "",
        opts.get("base_url") or model.base_url or "",
        options,
    )


async def vertex_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return await vertex_stream(model, context, options)


__all__ = ["google_vertex_stream", "vertex_stream", "vertex_stream_simple"]
