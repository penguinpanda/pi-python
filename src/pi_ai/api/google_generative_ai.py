"""Google Generative AI API 实现（对齐 TS packages/ai/src/api/google-generative-ai.ts）。

通过 Gemini REST streamGenerateContent + SSE 消费流式响应，转换为统一事件流。
"""

from __future__ import annotations

import asyncio
import json
import re

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
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    now_ms,
)
from ..utils._background import track_background_task
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.cost import calculate_cost
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import resolve_json_schema_strict_sampling
from ._shared import build_error_message, empty_usage
from .simple_options import clamp_max_tokens_to_context

_AsyncClient = httpx.AsyncClient
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _get_gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    return int(match.group(1)) if match is not None else None


def _supports_google_strict_tool_sampling(model_id: str) -> bool:
    version = _get_gemini_major_version(model_id)
    return version is not None and version >= 3


def _map_tool_choice(tool_choice: str) -> str:
    return {
        "auto": "AUTO",
        "none": "NONE",
        "any": "ANY",
    }.get(tool_choice, "AUTO")


def _resolve_google_function_calling_mode(
    tools: list[Tool],
    tool_choice: str | None,
    supports_strict: bool,
) -> str | None:
    use_strict = any(
        resolve_json_schema_strict_sampling(tool, supports_strict) is True for tool in tools
    )
    if tool_choice in ("none", "any"):
        return _map_tool_choice(tool_choice)
    if use_strict:
        return "VALIDATED"
    return _map_tool_choice(tool_choice) if tool_choice else None


def _is_gemma4_model(model_id: str) -> bool:
    return re.search(r"gemma-?4", model_id.lower()) is not None


def _is_gemini3_pro_model(model_id: str) -> bool:
    return re.match(r"gemini-3(?:\.\d+)?-pro", model_id.lower()) is not None


def _is_gemini3_flash_model(model_id: str) -> bool:
    lower = model_id.lower()
    return re.match(r"gemini-3(?:\.\d+)?-flash", lower) is not None or lower in (
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    )


def _google_disabled_thinking_config(model: Model) -> dict[str, Any]:
    if _is_gemini3_pro_model(model.id):
        return {"thinkingLevel": "LOW"}
    if _is_gemini3_flash_model(model.id):
        return {"thinkingLevel": "MINIMAL"}
    if _is_gemma4_model(model.id):
        return {"thinkingLevel": "MINIMAL"}
    return {"thinkingBudget": 0}


def _google_thinking_level(model: Model, effort: str) -> str:
    if _is_gemini3_pro_model(model.id):
        return "LOW" if effort in ("minimal", "low") else "HIGH"
    if _is_gemma4_model(model.id):
        return "MINIMAL" if effort in ("minimal", "low") else "HIGH"
    return {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }.get(effort, "HIGH")


def _get_google_budget(model: Model, effort: str, budgets: dict[str, int] | None) -> int:
    if budgets is not None and budgets.get(effort) is not None:
        return int(budgets[effort])
    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[effort]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]
    return -1


async def read_google_sse(
    bytes_iter: AsyncIterator[bytes],
) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    async for chunk in bytes_iter:
        buffer += chunk.decode("utf-8", errors="replace")
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            data = _extract_data(raw)
            if data:
                yield json.loads(data)
    if buffer.strip():
        data = _extract_data(buffer)
        if data:
            yield json.loads(data)


def _extract_data(raw: str) -> str | None:
    for line in raw.split("\n"):
        if line.startswith("data:"):
            value = line[5:].strip()
            return value if value and value != "[DONE]" else None
    return None


def _gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def _requires_tool_call_id(model_id: str) -> bool:
    """Google API 网关模型需要显式工具调用 ID（对齐 TS google-shared.ts requiresToolCallId）。

    Gemini 3+ / Claude-on-Vertex / gpt-oss 模型要求在 functionCall/functionResponse
    中携带 id，否则工具调用被拒。
    """
    gemini_major = _gemini_major_version(model_id)
    return (
        model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
        or (gemini_major is not None and gemini_major >= 3)
    )


def _normalize_tool_call_id(tool_call_id: str) -> str:
    """归一化工具调用 ID（对齐 TS normalizeToolCallId：仅保留 [A-Za-z0-9_-]，截 64 字符）。"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def _is_valid_thought_signature(signature: str | None) -> bool:
    """合法 base64 签名校验（对齐 TS isValidThoughtSignature）。"""
    if not signature or len(signature) % 4 != 0:
        return False
    return bool(re.match(r"^[A-Za-z0-9+/]+={0,2}$", signature))


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: Any) -> str | None:
    """仅同 provider/model 且 base64 合法时保留签名（对齐 TS resolveThoughtSignature）。"""
    if not is_same_provider_and_model or not isinstance(signature, str):
        return None
    return signature if _is_valid_thought_signature(signature) else None


def _to_google_contents(context: Context, model: Any) -> list[dict[str, Any]]:
    model_id: str = model.id if hasattr(model, "id") else str(model)
    model_provider: str | None = getattr(model, "provider", None)
    include_id = _requires_tool_call_id(model_id)
    contents: list[dict[str, Any]] = []
    for msg in context.messages:
        role = msg["role"]
        if role == "user":
            content = cast(Any, msg["content"])
            user_parts: list[dict[str, Any]]
            if isinstance(content, str):
                user_parts = [{"text": sanitize_surrogates(content)}]
            else:
                user_parts = []
                for block in content:
                    if block["type"] == "text":
                        user_parts.append({"text": sanitize_surrogates(block["text"])})
                    elif block["type"] == "image" and block.get("data"):
                        user_parts.append(
                            {
                                "inlineData": {
                                    "mimeType": block.get("mime_type") or "image/png",
                                    "data": block["data"],
                                }
                            }
                        )
            if user_parts:
                contents.append({"role": "user", "parts": user_parts})
        elif role == "assistant":
            assistant_msg = cast(Any, msg)
            # 仅同 provider/model 保留思考签名（对齐 TS convertMessages）。
            is_same_provider_and_model = (
                assistant_msg.get("provider") == model_provider
                and assistant_msg.get("model") == model_id
            )
            assistant_parts: list[dict[str, Any]] = []
            for block in assistant_msg.get("content") or []:
                block_type = block.get("type")
                if block_type == "text":
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model, block.get("text_signature")
                    )
                    text = block.get("text") or ""
                    # 空文本块仅在携带签名时保留（Gemini 要求回显签名，
                    # 否则思考链中断、任务中途停止）。
                    if not text.strip() and not thought_signature:
                        continue
                    part: dict[str, Any] = {"text": sanitize_surrogates(text)}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    assistant_parts.append(part)
                elif block_type == "thinking":
                    thinking = block.get("thinking") or ""
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(
                            is_same_provider_and_model,
                            block.get("thinking_signature"),
                        )
                        if not thinking.strip() and not thought_signature:
                            continue
                        thinking_part: dict[str, Any] = {
                            "thought": True,
                            "text": sanitize_surrogates(thinking),
                        }
                        if thought_signature:
                            thinking_part["thoughtSignature"] = thought_signature
                    else:
                        # 跨 provider/model：签名不可用，思考转纯文本（不带标签）。
                        if not thinking.strip():
                            continue
                        thinking_part = {"text": sanitize_surrogates(thinking)}
                    assistant_parts.append(thinking_part)
                elif block_type == "toolCall":
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model, block.get("thought_signature")
                    )
                    function_call = {
                        "name": block["name"],
                        "args": block.get("arguments") or {},
                    }
                    if include_id and block.get("id"):
                        function_call["id"] = _normalize_tool_call_id(block["id"])
                    tool_part: dict[str, Any] = {"functionCall": function_call}
                    if thought_signature:
                        tool_part["thoughtSignature"] = thought_signature
                    assistant_parts.append(tool_part)
            if assistant_parts:
                contents.append({"role": "model", "parts": assistant_parts})
        elif role == "toolResult":
            tool_msg = cast(Any, msg)
            text = "".join(
                block.get("text") or "" for block in tool_msg["content"] if block["type"] == "text"
            )
            function_response: dict[str, Any] = {
                "name": tool_msg["tool_name"],
                "response": {"error" if tool_msg.get("is_error") else "output": text},
            }
            if include_id and tool_msg.get("tool_call_id"):
                function_response["id"] = _normalize_tool_call_id(tool_msg["tool_call_id"])
            tool_parts = [{"functionResponse": function_response}]
            contents.append({"role": "user", "parts": tool_parts})
    return contents


def _to_google_tools(context: Context, use_parameters: bool = False) -> list[dict[str, Any]] | None:
    if not context.tools:
        return None
    # 对齐 TS convertTools：默认 parametersJsonSchema（完整 JSON Schema），
    # use_parameters=True 时使用 legacy parameters（OpenAPI 3.03）。
    declarations = []
    for tool in context.tools:
        declaration: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }
        if use_parameters:
            declaration["parameters"] = tool.input_schema
        else:
            declaration["parametersJsonSchema"] = tool.input_schema
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in (
        "SAFETY",
        "RECITATION",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "IMAGE_SAFETY",
    ):
        return "error"
    return "stop"


def google_generative_ai_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    opts = options or {}
    contents = _to_google_contents(context, model)
    tools = _to_google_tools(context)

    config: dict[str, Any] = {}
    temperature = opts.get("temperature")
    if temperature is not None:
        config["temperature"] = temperature
    requested = opts.get("max_tokens")
    if requested is not None:
        config["maxOutputTokens"] = clamp_max_tokens_to_context(model, context, requested)
    reasoning = opts.get("reasoning")
    thinking_enabled = opts.get("thinking_enabled")
    if thinking_enabled or (reasoning is not None and reasoning != "off"):
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        effort = "high" if reasoning in (None, "off", "xhigh", "max") else str(reasoning)
        if (
            _is_gemini3_pro_model(model.id)
            or _is_gemini3_flash_model(model.id)
            or _is_gemma4_model(model.id)
        ):
            thinking_config["thinkingLevel"] = _google_thinking_level(model, effort)
        else:
            budget_value = opts.get("thinking_budget")
            if budget_value is not None:
                thinking_config["thinkingBudget"] = int(budget_value)
            else:
                budget = _get_google_budget(
                    model,
                    effort,
                    cast(dict[str, int] | None, opts.get("thinking_budgets")),
                )
                if budget >= 0:
                    thinking_config["thinkingBudget"] = budget
        config["thinkingConfig"] = thinking_config
    elif model.reasoning:
        config["thinkingConfig"] = _google_disabled_thinking_config(model)

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": config,
    }
    if context.system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": context.system_prompt}]}
    if tools:
        payload["tools"] = tools
        mode = _resolve_google_function_calling_mode(
            context.tools or [],
            cast(str | None, opts.get("tool_choice")),
            _supports_google_strict_tool_sampling(model.id),
        )
        if mode is not None:
            payload["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

    async def _run() -> None:
        content_blocks: list[ContentBlock] = []
        current_index: int | None = None
        current_kind: str | None = None
        usage: Usage = empty_usage()
        stop_reason: StopReason = "stop"
        response_id: str | None = None
        has_tool_call = False
        client: Any = None
        response: Any = None

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
            current_kind = None
            current_index = None

        try:
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            base = (base_url or model.base_url or _DEFAULT_BASE_URL).rstrip("/")
            url = f"{base}/models/{model.id}:streamGenerateContent?alt=sse"
            headers: dict[str, str] = {
                "content-type": "application/json",
                "accept": "text/event-stream",
            }
            if api_key:
                headers["x-goog-api-key"] = api_key
            for name, value in (opts.get("headers") or {}).items():
                if value is not None:
                    headers[name] = value

            timeout_ms = opts.get("timeout_ms") or 120000

            class _GoogleApiError(RuntimeError):
                """携带 status/headers 的 Google API 错误（retry_provider_request 判定用）。"""

                def __init__(self, status_code: int, headers: Any, message: str) -> None:
                    super().__init__(message)
                    self.status = status_code
                    self.headers = headers

            async def _open_stream():
                client = _AsyncClient(timeout=timeout_ms / 1000)
                try:
                    response = await client.send(
                        client.build_request("POST", url, headers=headers, json=payload),
                        stream=True,
                    )
                except BaseException:
                    await client.aclose()
                    raise
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    await client.aclose()
                    raise _GoogleApiError(
                        response.status_code,
                        dict(response.headers),
                        f"Gemini API error ({response.status_code}): {body}",
                    )
                return client, response

            # 对齐 TS retryGoogleRequest：408/409/429/5xx 指数退避重试（可中断）。
            max_retries = opts.get("maxRetries")
            client, response = await retry_provider_request(
                _open_stream,
                max_retries=int(max_retries) if isinstance(max_retries, (int, float)) else 0,
                max_retry_delay_ms=cast(int | None, opts.get("maxRetryDelayMs")),
                signal=opts.get("signal"),
            )
            stream.push(StartEvent(type="start", partial=_partial()))
            async for chunk in read_google_sse(response.aiter_bytes()):
                response_id = response_id or chunk.get("responseId")
                candidate = (chunk.get("candidates") or [{}])[0]
                if candidate.get("finishReason"):
                    stop_reason = _map_stop_reason(candidate["finishReason"])
                usage_metadata = chunk.get("usageMetadata") or {}
                if usage_metadata:
                    usage = empty_usage()
                    cached = int(usage_metadata.get("cachedContentTokenCount") or 0)
                    usage["input"] = int(usage_metadata.get("promptTokenCount") or 0) - cached
                    usage["output"] = int(usage_metadata.get("candidatesTokenCount") or 0) + int(
                        usage_metadata.get("thoughtsTokenCount") or 0
                    )
                    usage["cache_read"] = cached
                    usage["reasoning"] = int(usage_metadata.get("thoughtsTokenCount") or 0)
                    usage["total_tokens"] = int(usage_metadata.get("totalTokenCount") or 0)
                    calculate_cost(model, usage)
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if "text" in part:
                        is_thinking = bool(part.get("thought"))
                        if is_thinking:
                            if current_kind != "thinking":
                                _end_current_block()
                                current_kind = "thinking"
                                content_blocks.append(
                                    ThinkingContent(
                                        type="thinking",
                                        thinking="",
                                        thinking_signature=part.get("thoughtSignature"),
                                    )
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
                            text = part["text"]
                            thinking_block["thinking"] += text
                            stream.push(
                                ThinkingDeltaEvent(
                                    type="thinking_delta",
                                    content_index=index,
                                    delta=text,
                                    partial=_partial(),
                                )
                            )
                        else:
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
                            text = part["text"]
                            text_block["text"] += text
                            stream.push(
                                TextDeltaEvent(
                                    type="text_delta",
                                    content_index=index,
                                    delta=text,
                                    partial=_partial(),
                                )
                            )
                    if "functionCall" in part:
                        _end_current_block()
                        call = part["functionCall"]
                        tool_block = ToolCall(
                            type="toolCall",
                            id=call.get("id") or f"{call.get('name') or 'tool'}_{now_ms()}",
                            name=call.get("name") or "",
                            raw_arguments=json.dumps(call.get("args") or {}),
                            arguments=call.get("args") or {},
                        )
                        content_blocks.append(tool_block)
                        current_index = len(content_blocks) - 1
                        current_kind = "toolCall"
                        has_tool_call = True
                        stream.push(
                            ToolCallStartEvent(
                                type="toolcall_start",
                                content_index=current_index,
                                partial=_partial(),
                            )
                        )
                        stream.push(
                            ToolCallDeltaEvent(
                                type="toolcall_delta",
                                content_index=current_index,
                                delta=tool_block["raw_arguments"],
                                partial=_partial(),
                            )
                        )
                        stream.push(
                            ToolCallEndEvent(
                                type="toolcall_end",
                                content_index=current_index,
                                tool_call=tool_block,
                                partial=_partial(),
                            )
                        )
            _end_current_block()
            if has_tool_call and stop_reason != "error":
                stop_reason = "tool_call"
            if stop_reason == "error":
                raise RuntimeError(f"Gemini stream stopped with: {candidate.get('finishReason')}")
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
                response_id=response_id or "",
            )
            assert stop_reason in ("stop", "length", "tool_call")
            stream.push({"type": "done", "reason": stop_reason, "message": msg})
        except asyncio.CancelledError:
            stream.error(asyncio.CancelledError())
            raise
        except Exception as exc:
            err_msg = build_error_message(model, exc)
            stream.push({"type": "error", "reason": "error", "error": err_msg})
        finally:
            # 取消/异常路径也必须关闭流式响应与客户端（httpx 无 __del__，
            # 否则每次中止都泄漏一个持有连接池的 AsyncClient）。
            if response is not None:
                try:
                    await response.aclose()
                except Exception:
                    pass
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass

    track_background_task(_run())
    return stream


def google_stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    opts = options or {}
    return google_generative_ai_stream(
        model,
        context,
        opts.get("api_key") or "",
        opts.get("base_url") or model.base_url or "",
        options,
    )


def google_stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    return google_stream(model, context, options)


__all__ = [
    "google_generative_ai_stream",
    "google_stream",
    "google_stream_simple",
    "read_google_sse",
]
