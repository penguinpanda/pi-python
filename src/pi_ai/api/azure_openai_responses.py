"""Azure OpenAI Responses API（复用 responses.py 的解析管线）。

通过 client_factory 注入 AsyncAzureOpenAI，并传递 deployment 名称作为请求模型。
"""

from __future__ import annotations

from typing import Any, Callable, cast
from urllib.parse import urlparse, urlunparse

from openai import AsyncAzureOpenAI

from ..types import (
    Context,
    Model,
    StreamOptions,
)
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from .responses import responses_stream

_DEFAULT_API_VERSION = "2024-10-21"


def _azure_root(base_url: str) -> str:
    parsed = urlparse(base_url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _resolve_endpoint(model: Model, options: dict[str, Any]) -> str:
    base_url = (
        options.get("azure_base_url")
        or get_provider_env_value("AZURE_OPENAI_BASE_URL", options.get("env"))
        or model.base_url
        or ""
    )
    resource_name = (
        options.get("azure_resource_name")
        or get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", options.get("env"))
        or ""
    )
    if not base_url and resource_name:
        base_url = f"https://{resource_name}.openai.azure.com"
    if not base_url:
        raise ValueError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or "
            "AZURE_OPENAI_RESOURCE_NAME."
        )
    return _azure_root(base_url.rstrip("/"))


def _resolve_deployment(model: Model, options: dict[str, Any]) -> str:
    explicit = options.get("azure_deployment_name")
    if explicit:
        return str(explicit)
    mapping = get_provider_env_value(
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        options.get("env"),
    )
    if mapping:
        for entry in mapping.split(","):
            model_id, _, deployment = entry.strip().partition("=")
            if model_id == model.id and deployment:
                return deployment.strip()
    return model.id


def _resolve_api_version(options: dict[str, Any]) -> str:
    return (
        str(options.get("azure_api_version") or "")
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", options.get("env"))
        or _DEFAULT_API_VERSION
    )


async def azure_openai_responses_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts: dict[str, Any] = dict(options or {})
    if base_url:
        opts["azure_base_url"] = base_url
    endpoint = _resolve_endpoint(model, opts)
    deployment = _resolve_deployment(model, opts)
    api_version = _resolve_api_version(opts)
    headers = dict(model.headers or {})
    for name, value in (opts.get("headers") or {}).items():
        if value is not None:
            headers[name] = value

    def _factory(
        _api_key: str,
        _base_url: str,
        *,
        timeout: float,
        max_retries: int,
        headers: dict[str, str] | None,
    ) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            api_key=_api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=headers,
        )

    return await responses_stream(
        model,
        context,
        api_key,
        base_url,
        options,
        client_factory=cast(Callable[..., Any], _factory),
        request_model_id=deployment,
    )


async def azure_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    return await azure_openai_responses_stream(
        model,
        context,
        opts.get("api_key") or "",
        opts.get("base_url") or model.base_url or "",
        options,
    )


async def azure_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return await azure_stream(model, context, options)


__all__ = ["azure_openai_responses_stream", "azure_stream", "azure_stream_simple"]
