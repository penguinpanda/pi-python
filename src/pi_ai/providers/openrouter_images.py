"""OpenRouter 图片生成（对齐 TS api/openrouter-images.ts）。

OpenRouter 图片生成走 Chat Completions 接口的 modalities 扩展：
请求携带 modalities: ["image", "text"]，响应 message.images[] 为
data URI；本实现永不 reject（失败返回 stopReason=error 的 AssistantImages）。
"""

import asyncio
import inspect
import re

from typing import Any

import httpx

from ..types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    Usage,
    now_ms,
)

_AsyncClient = httpx.AsyncClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _build_params(model: ImagesModel, context: ImagesContext) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in context.get("input", []):
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item['mime_type']};base64,{item['data']}"},
                }
            )
    modalities = ["image", "text"] if "text" in model.output else ["image"]
    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": modalities,
    }


def _parse_usage(raw: dict[str, Any], model: ImagesModel) -> Usage:
    prompt_tokens = int(raw.get("prompt_tokens", 0) or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    cache_write = int(details.get("cache_write_tokens", 0) or 0)
    cache_read = max(0, cached - cache_write)
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    output_tokens = int(raw.get("completion_tokens", 0) or 0)
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )


def _image_url_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        url = value.get("url")
        return url if isinstance(url, str) else None
    return None


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions | None = None,
) -> AssistantImages:
    output: AssistantImages = {
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "output": [],
        "stop_reason": "stop",
        "timestamp": now_ms(),
    }
    opts = options or {}
    try:
        api_key = opts.get("api_key")
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        params = _build_params(model, context)
        on_payload = opts.get("on_payload")
        if on_payload is not None:
            next_params = on_payload(params, model)
            if inspect.isawaitable(next_params):
                next_params = await next_params
            if next_params is not None:
                params = next_params

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        request_headers = opts.get("headers") or {}
        for name, value in request_headers.items():
            if value is not None:
                headers[name] = value

        timeout_ms = opts.get("timeout_ms") or 120000
        async with _AsyncClient(timeout=timeout_ms / 1000) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=params,
            )

        on_response = opts.get("on_response")
        if on_response is not None:
            event = {"status": response.status_code, "headers": dict(response.headers)}
            result = on_response(event, model)
            if inspect.isawaitable(result):
                await result

        response.raise_for_status()
        data = response.json()
        output["response_id"] = data.get("id")
        if isinstance(data.get("usage"), dict):
            output["usage"] = _parse_usage(data["usage"], model)

        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                output["output"].append({"type": "text", "text": content})
            for image in message.get("images") or []:
                if not isinstance(image, dict):
                    continue
                url_value = _image_url_value(image.get("image_url"))
                if not url_value or not url_value.startswith("data:"):
                    continue
                match = _DATA_URI_RE.match(url_value)
                if not match:
                    continue
                output["output"].append(
                    {
                        "type": "image",
                        "url": None,
                        "mime_type": match.group(1),
                        "data": match.group(2),
                    }
                )
        return output
    except asyncio.CancelledError:
        output["stop_reason"] = "aborted"
        return output
    except Exception as exc:
        output["stop_reason"] = (
            "aborted" if opts.get("signal") is not None and opts["signal"].is_set() else "error"
        )
        output["error_message"] = str(exc)
        return output


__all__ = ["generate_images", "OPENROUTER_BASE_URL"]
