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
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _resolve_adc_credentials(
    options: dict[str, Any],
) -> tuple[str | None, str | None]:
    """从 Application Default Credentials 解析 (token, project)；未安装 google-auth 时返回 (None, None)。"""
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception:
        return None, None
    try:
        credentials, project = google.auth.default(
            scopes=[_CLOUD_PLATFORM_SCOPE],
            request=Request(),
        )
        credentials.refresh(Request())
        return getattr(credentials, "token", None), project
    except Exception:
        return None, None


def _is_vertex_api_key_auth(api_key: str, options: dict[str, Any]) -> bool:
    """区分 Cloud API key（x-goog-api-key）与 OAuth access token（Bearer）。"""
    if options.get("vertex_auth_kind") == "api_key":
        return True
    cloud_api_key = get_provider_env_value("GOOGLE_CLOUD_API_KEY", options.get("env"))
    return bool(api_key and cloud_api_key and api_key == cloud_api_key)


def _resolve_vertex_options(
    model: Model,
    api_key: str,
    base_url: str,
    options: dict[str, Any],
) -> tuple[str, str]:
    env = options.get("env")
    api_key_kind = _is_vertex_api_key_auth(api_key, options)
    token = (
        ""
        if api_key_kind
        else (api_key or get_provider_env_value("GOOGLE_OAUTH_ACCESS_TOKEN", env))
    )
    project = (
        options.get("project")
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", env)
        or get_provider_env_value("GCP_PROJECT", env)
    )
    if not api_key_kind and not token:
        token, adc_project = _resolve_adc_credentials(options)
        if project is None:
            project = adc_project
    if not token and not api_key_kind:
        raise RuntimeError(
            f"No access token for provider: {model.provider}. "
            "Set GOOGLE_OAUTH_ACCESS_TOKEN or configure Application Default Credentials "
            "(google-auth + GOOGLE_APPLICATION_CREDENTIALS / gcloud auth application-default login)."
        )
    if not project:
        raise RuntimeError("Google Vertex requires a project")
    location = (
        options.get("location")
        or get_provider_env_value("GOOGLE_CLOUD_LOCATION", env)
        or _DEFAULT_LOCATION
    )
    endpoint = (
        base_url
        or f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/publishers/google"
    )
    return cast(str, api_key if api_key_kind else token), endpoint


async def google_vertex_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = dict(options or {})
    value, endpoint = _resolve_vertex_options(model, api_key, base_url, opts)
    auth_kind = "api_key" if _is_vertex_api_key_auth(api_key, opts) else "token"
    headers = dict(cast(dict[str, str | None] | None, opts.get("headers")) or {})
    if auth_kind == "token":
        headers["Authorization"] = f"Bearer {value}"
        value_for_google = ""
    else:
        value_for_google = value
    merged = dict(opts)
    merged["headers"] = headers
    return google_generative_ai_stream(
        model,
        context,
        value_for_google,
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
