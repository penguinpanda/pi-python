"""Provider HTTP 错误对象归一化（对齐 TS utils/error-body.ts）。

网关/proxy 返回的非 2xx 响应体常藏在 SDK 特有字段里（statusCode/status/
$metadata.httpStatusCode/$response.body），只读 error.message 的 catch 块会
丢失 body。normalize_provider_error 探测已知 SDK 字段形状并返回统一结构。
"""

from __future__ import annotations

import json
from typing import Any

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


def normalize_provider_error(error: Any) -> dict[str, Any]:
    """归一化 provider 错误（对齐 TS normalizeProviderError）。"""
    if not isinstance(error, BaseException):
        return {
            "message": _safe_json_stringify(error),
            "messageCarriesBody": False,
        }

    status = _extract_status(error)
    body = _extract_body(error)
    message = str(error)
    message_carries_body = body is None or (body in message)

    result: dict[str, Any] = {
        "message": message,
        "messageCarriesBody": message_carries_body,
    }
    if status is not None:
        result["status"] = status
    if body is not None:
        result["body"] = body
    return result


def _extract_status(error: BaseException) -> int | None:
    """按 SDK 字段顺序探测 HTTP 状态码：statusCode → status →
    $metadata.httpStatusCode → $response.statusCode（对齐 TS extractStatus）。"""
    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "statusCode", None),
        getattr(error, "status", None),
    )
    for value in candidates:
        if isinstance(value, int):
            return value
    metadata = getattr(error, "$metadata", None)
    if metadata is not None:
        value = getattr(metadata, "httpStatusCode", None)
        if isinstance(value, int):
            return value
    response = getattr(error, "$response", None)
    if response is not None:
        value = getattr(response, "statusCode", None)
        if isinstance(value, int):
            return value
    return None


def _extract_body(error: BaseException) -> str | None:
    """探测错误体并按 4000 字符上限截断（对齐 TS extractBody）。"""
    raw = _probe_body(error)
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        text = json.dumps(raw, ensure_ascii=False)
    else:
        text = str(raw)
    return text[:MAX_PROVIDER_ERROR_BODY_CHARS]


def _probe_body(error: BaseException) -> Any:
    body = getattr(error, "body", None)
    if body is not None:
        return body
    nested = getattr(error, "error", None)
    if nested is not None:
        nested_body = getattr(nested, "body", None) if not isinstance(nested, dict) else nested
        if nested_body is not None:
            return nested_body
    response = getattr(error, "$response", None)
    if response is not None:
        return getattr(response, "body", None)
    return None


def _safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["MAX_PROVIDER_ERROR_BODY_CHARS", "normalize_provider_error"]
