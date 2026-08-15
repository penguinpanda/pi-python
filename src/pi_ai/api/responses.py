"""
OpenAI Responses API 实现。

=========================================================
模块职责
=========================================================

本模块负责调用 OpenAI Responses API，

并将其事件(Event)流转换为 SDK 内部事件流。

整个处理流程如下：

        Context
            │
            ▼
    _to_responses_input()

            │
            ▼
OpenAI Responses API

            │
            ▼
Responses Event Stream

            │
            ▼
转换为 SDK Event

            │
            ▼
AssistantMessageEventStream

与 Chat Completions API 不同，

Responses API 返回的是"事件(Event)"，

例如：

    response.output_text.delta

    response.function_call_arguments.delta

    response.completed

因此本模块本质上是：

    Responses Event

        ↓

    SDK Event

的适配层(Adapter)。
"""

import asyncio
import json
from typing import Any, Callable, cast
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from ..utils._background import track_background_task
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.cost import calculate_cost
from ..types import (
    AssistantMessage,
    ContentBlock,
    Context,
    Model,
    ModelThinkingLevel,
    StartEvent,
    StopReason,
    StreamOptions,
    Tool,
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
from ._shared import (
    build_error_message,
    close_async_client,
    empty_usage,
    parse_tool_arguments,
    to_responses_tools,
)
from .simple_options import clamp_max_tokens_to_context
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .compat_runtime import (
    compat_value,
    supports_long_cache_retention,
    supports_strict_mode,
)
from .transform_messages import (
    normalize_responses_tool_call_id,
    short_hash,
    transform_messages,
)
from ..utils.prompt_cache import (
    clamp_openai_prompt_cache_key,
    resolve_cache_retention,
)
from ..utils.deferred_tools import split_deferred_tools


def encode_text_signature_v1(id_: str, phase: str | None = None) -> str:
    """构造 TextSignatureV1 JSON（对齐 TS encodeTextSignatureV1）。

    用于把 Responses message item 的 id / phase 持久化到
    TextContent.text_signature，供后续轮次回放。
    """

    payload: dict[str, Any] = {"v": 1, "id": id_}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload, ensure_ascii=False)


def parse_text_signature(signature: str | None) -> dict[str, Any] | None:
    """解析 TextContent.text_signature（对齐 TS parseTextSignature）。

    - 以 "{" 开头：尝试 JSON 解析，v==1 且 id 为 str 时返回 {id, phase?}
    - 其他（旧版纯字符串 id）：返回 {id: signature}
    """

    if not signature:
        return None

    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            phase = parsed.get("phase")
            result: dict[str, Any] = {"id": parsed["id"]}
            if phase in ("commentary", "final_answer"):
                result["phase"] = phase
            return result

    return {"id": signature}


def _to_jsonable(obj: Any) -> Any:
    """递归转换为可 JSON 序列化的结构。

    兼容 openai SDK 的 pydantic 模型与测试用的 SimpleNamespace。
    """

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _convert_tool_result_output(
    model: Model | None,
    content: list[Any],
) -> str | list[dict[str, Any]]:
    """转换 toolResult 内容（对齐 TS convertToolResultOutput）。

    - 非视觉模型 / 无图片：返回纯文本（空结果用占位文本）。
    - 视觉模型且含图片：返回 input_text + input_image 数组。
    """

    text_result = "\n".join(b["text"] for b in content if b.get("type") == "text")
    images = [b for b in content if b.get("type") == "image"]
    has_text = len(text_result) > 0

    if not images or model is None or not (model.input and "image" in model.input):
        if has_text:
            return text_result
        return "(see attached image)" if images else "(no tool output)"

    output: list[dict[str, Any]] = []
    if has_text:
        output.append({"type": "input_text", "text": text_result})
    for image in images:
        mime = image.get("mime_type") or "image/png"
        output.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime};base64,{image.get('data') or ''}",
            }
        )
    return output


def _create_client(
    api_key: str,
    base_url: str = "",
    timeout: float = 180.0,
    max_retries: int = 2,
    headers: dict[str, str | None] | None = None,
) -> AsyncOpenAI:
    """
    创建 AsyncOpenAI 客户端。

    统一配置：

        • API Key
        • Base URL
        • Timeout
        • Retry（默认 2，可从 StreamOptions.max_retries 覆盖）

    Responses API 与 Chat Completions API

    共用 AsyncOpenAI 客户端。
    """

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": httpx.Timeout(timeout),
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if headers:
        kwargs["default_headers"] = {k: v for k, v in headers.items() if v is not None}

    return AsyncOpenAI(**kwargs)


def _is_official_deepseek_base_url(base_url: str) -> bool:
    """DeepSeek 官方域名（api.deepseek.com 及 *.deepseek.com）。"""
    host = urlparse(base_url).hostname or ""
    return host == "api.deepseek.com" or host.endswith(".deepseek.com")


def _resolve_web_search(
    model: Model,
    base_url: str,
    options: StreamOptions | None,
) -> bool:
    """解析是否启用服务端 web_search。

    显式 StreamOptions.web_search 优先；未显式时仅在官方 DeepSeek
    Responses 且模型 compat 声明支持时默认开启。
    """
    opts = options or {}
    if opts.get("web_search") is not None:
        return bool(opts["web_search"])
    if not compat_value(model, "supportsWebSearch", False):
        return False
    return model.provider == "deepseek" and _is_official_deepseek_base_url(base_url)


def _build_responses_reasoning(
    model: Model,
    reasoning_level: ModelThinkingLevel | None,
    reasoning_summary: str | None = None,
) -> dict[str, Any] | None:
    """把 pi 推理级别翻译为 Responses API 的 reasoning 参数。"""
    if reasoning_level is None:
        return None

    mapping = model.thinking_level_map or {}
    if reasoning_level == "off":
        if mapping.get("off") is None and "off" in mapping:
            return None
        return {"effort": "none"}

    if reasoning_level in mapping:
        mapped = mapping[reasoning_level]
        if mapped is None:
            return None
        effort = mapped
    else:
        effort = reasoning_level
    result: dict[str, Any] = {"effort": effort}
    result["summary"] = reasoning_summary or "auto"
    return result


def _build_responses_request_kwargs(
    model: Model,
    context: Context,
    base_url: str,
    options: StreamOptions | None,
    *,
    request_model_id: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """构造 Responses API 请求参数（含 web_search 开关，供调用方复用）。

    与 Codex WebSocket 路径共享：WS 需要先于流启动建立连接（连接失败
    才能同步回退 SSE），而连接需要完整的请求体。
    """
    opts = options or {}
    web_search_enabled = _resolve_web_search(model, base_url, options)
    tool_search_enabled = compat_value(model, "supportsToolSearch", False)
    immediate_tools, deferred_tools = split_deferred_tools(context, tool_search_enabled)
    transformed_messages = transform_messages(
        context.messages, model, normalize_responses_tool_call_id
    )
    input_items = _to_responses_input(
        transformed_messages,
        model,
        replay_web_search_items=web_search_enabled,
        deferred_tools=deferred_tools,
    )
    tools: list[dict[str, Any]] = []
    if web_search_enabled:
        tools.append({"type": "web_search"})
    if immediate_tools:
        tools.extend(
            to_responses_tools(
                immediate_tools,
                supports_strict_mode=supports_strict_mode(model),
            )
        )

    kwargs: dict[str, Any] = {
        "model": request_model_id or model.id,
        "input": input_items,
        "stream": True,
        "store": False,
    }
    include_system_prompt = opts.get("include_system_prompt", True)
    if include_system_prompt and context.system_prompt:
        kwargs["instructions"] = context.system_prompt
    if tools:
        kwargs["tools"] = tools
    temperature = opts.get("temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature
    # max_output_tokens 收敛到模型上下文窗口内（对齐 TS buildBaseOptions）：
    # 未指定时使用模型默认 max_tokens，始终发送收敛后的值。
    requested = opts.get("max_tokens")
    kwargs["max_output_tokens"] = clamp_max_tokens_to_context(
        model,
        context,
        requested if requested is not None else model.max_tokens,
    )
    reasoning = _build_responses_reasoning(
        model,
        opts.get("reasoning"),
        opts.get("reasoning_summary"),
    )
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
        if reasoning.get("effort") not in (None, "none"):
            kwargs["include"] = ["reasoning.encrypted_content"]
    elif model.provider == "xai" and model.reasoning:
        kwargs["include"] = ["reasoning.encrypted_content"]

    # 提示缓存（Prompt Cache，对齐 TS openai-responses.ts）：
    #   prompt_cache_key：仅 supportsExplicitPromptCacheMode 时发送
    #   prompt_cache_retention：long 且支持长缓存时发送 "24h"
    if compat_value(model, "supportsExplicitPromptCacheMode", True):
        cache_retention = resolve_cache_retention(opts.get("cache_retention"), opts.get("env"))
        supports_long = supports_long_cache_retention(model)
        if cache_retention != "none":
            kwargs["prompt_cache_key"] = clamp_openai_prompt_cache_key(opts.get("session_id"))
        if cache_retention == "long" and supports_long:
            kwargs["prompt_cache_retention"] = "24h"
    return kwargs, web_search_enabled


def _parse_response_usage(resp: Any, model: Model) -> Usage:
    """从 Responses response 提取 usage，包含缓存与推理 token 明细。"""
    usage_obj = getattr(resp, "usage", None)
    if not usage_obj:
        return empty_usage()

    input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
    total_tokens = getattr(usage_obj, "total_tokens", 0) or 0
    input_details = getattr(usage_obj, "input_tokens_details", None)
    output_details = getattr(usage_obj, "output_tokens_details", None)
    cache_read = 0
    cache_write = 0
    reasoning_tokens = 0
    if input_details is not None:
        cache_read = getattr(input_details, "cached_tokens", 0) or 0
        cache_write = getattr(input_details, "cache_write_tokens", 0) or 0
    if output_details is not None:
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

    # OpenAI 的 input_tokens 已包含 cached / cache-write token：
    # 不扣减会导致缓存命中部分既按全价 input 计费、又按缓存价计费。
    input_tokens = max(0, input_tokens - cache_read - cache_write)
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=total_tokens or (input_tokens + output_tokens + cache_read + cache_write),
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )
    if reasoning_tokens:
        usage["reasoning"] = reasoning_tokens
    calculate_cost(model, usage)
    return usage


def _to_responses_input(
    messages: list[Any],
    model: Model | None = None,
    replay_web_search_items: bool = True,
    deferred_tools: dict[str, Tool] | None = None,
) -> list[dict[str, Any]]:
    """
    将 SDK Message

    转换为 Responses API Input。

    不同于 Chat Completions：

    Responses API

    使用 input 字段，

    格式也有所不同。

    例如：

    SDK

    ↓

    UserMessage

    ↓

    Responses Input Item

    同时负责：

        • System

        • User

        • Assistant History

        • Tool Result

    全部转换。

    Parameters
    ----------
    model
        可选的模型元数据。

        用于按模型能力过滤图片输入。

        当模型不支持 image 时，图片内容会被跳过。

        为 None 时不做过滤（保持向后兼容）。

    replay_web_search_items
        是否回放历史 assistant 消息中的 web_search_call 原始 item。
        服务端 web_search 关闭时传 False，避免把已失效的搜索项塞回 input。
    """
    items: list[dict[str, Any]] = []
    loaded_deferred_names: set[str] = set()

    # 消息索引：用于文本消息 fallback id（msg_pi_{index}）。
    msg_index = 0

    for msg in messages:
        role = msg["role"]

        if role == "system":
            # System Prompt。
            #
            # Responses API
            #
            # 可以在对话任意位置出现
            # System Message。
            items.append({"role": "system", "content": msg["content"]})

        # User Message。
        #
        # 支持：
        #
        # • 文本
        #
        # • 图片
        #
        # 转换成 Responses API
        #
        # input_text
        #
        # input_image。
        elif role == "user":
            content = msg["content"]
            if isinstance(content, str):
                items.append({"role": "user", "content": content})
            else:
                # Multi-part user content
                parts: list[dict[str, Any]] = []
                for block in content:
                    if block["type"] == "text":
                        parts.append({"type": "input_text", "text": block["text"]})
                    elif block["type"] == "image" and (
                        model is None or "image" in (model.input or [])
                    ):
                        img: dict[str, Any] = {"type": "input_image"}
                        if block.get("url"):
                            img["image_url"] = block["url"]
                        elif block.get("data"):
                            # Base64 图片
                            #
                            # 转换为 Data URL。
                            img["image_url"] = (
                                f"data:{block.get('mime_type', 'image/png')};base64,{block['data']}"
                            )
                        parts.append(img)
                items.append({"role": "user", "content": parts})

        # Assistant 历史消息。
        #
        # 对齐 TS openai-responses-shared：
        #
        # 每个内容块展开为独立的顶层 Response Item：
        #
        #     thinking → reasoning（回放 thinking_signature 中的完整 item）
        #     text     → message（output_text + id / phase）
        #     toolCall → function_call（call_id + fc_ item id）
        #
        # 而不是包在单个 {"role":"assistant","content":[...]} 里，
        # 因为 reasoning item 必须作为独立项存在。
        elif role == "assistant":
            output: list[dict[str, Any]] = []
            is_different_model = bool(
                model
                and msg.get("model") != model.id
                and msg.get("provider") == model.provider
                and msg.get("api") == model.api
            )
            text_block_index = 0

            # 服务端 web_search 的历史项在 stateless 模式下必须原样回传，
            # 否则后续轮次拿不到搜索结果上下文。
            if replay_web_search_items and not is_different_model:
                for item in msg.get("responses_items") or []:
                    if isinstance(item, dict) and item.get("type") == "web_search_call":
                        output.append(item)

            for block in msg["content"]:
                block_type = block["type"]

                # 推理历史 → reasoning item 回放。
                #
                # thinking_signature 存有完整的 ResponseReasoningItem
                # （含 id / summary / content / encrypted_content），
                # 原样回放以支持多轮续传（OpenAI 的 store:false 场景）。
                if block_type == "thinking":
                    signature = block.get("thinking_signature")
                    if signature:
                        try:
                            reasoning_item = json.loads(signature)
                        except (ValueError, TypeError):
                            reasoning_item = None
                        if isinstance(reasoning_item, dict):
                            output.append(reasoning_item)

                # 文本块 → message item。
                #
                # id 优先取 text_signature 中的持久化 id，
                # 否则用 msg_pi_{index} fallback；超过 64 字符用 short_hash。
                elif block_type == "text":
                    parsed = parse_text_signature(block.get("text_signature"))
                    fallback_id = (
                        f"msg_pi_{msg_index}"
                        if text_block_index == 0
                        else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1

                    msg_id = parsed["id"] if parsed else None
                    if not msg_id:
                        msg_id = fallback_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"

                    text_item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": block["text"],
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed and parsed.get("phase"):
                        text_item["phase"] = parsed["phase"]
                    output.append(text_item)

                # 工具调用 → function_call item。
                elif block_type == "toolCall":
                    call_id, _, item_id_raw = block["id"].partition("|")
                    item_id = item_id_raw or None

                    # 仅保留 fc_ 开头且非跨模型的 item id。
                    #
                    # OpenAI 会校验 fc_xxx 与 rs_xxx reasoning 的配对；
                    # 跨模型消息省略 id 可避开该校验（与跨 provider 一致）。
                    keep_item_id = (
                        item_id is not None and item_id.startswith("fc_") and not is_different_model
                    )
                    if not keep_item_id:
                        item_id = None

                    fc_item: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": block["name"],
                        "arguments": json.dumps(
                            block["arguments"] if block["arguments"] is not None else {},
                            ensure_ascii=False,
                        ),
                    }
                    if item_id is not None:
                        fc_item["id"] = item_id
                    output.append(fc_item)

            if not output:
                msg_index += 1
                continue
            items.extend(output)

        # Tool 调用结果。
        #
        # Responses API
        #
        # 使用 function_call_output。
        elif role == "toolResult":
            # 双段 ID（call_id|fc_item_id）只保留 call_id。
            call_id = msg["tool_call_id"].partition("|")[0]
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _convert_tool_result_output(model, msg["content"]),
                }
            )
            # Deferred tools：toolResult 里新增的工具通过 tool_search 协议
            # 以 client 执行方式补交定义（对齐 TS deferredTools 路径）。
            if deferred_tools:
                deferred_list: list[Tool] = []
                for name in msg.get("added_tool_names") or []:
                    tool = deferred_tools.get(name)
                    if tool is None or name in loaded_deferred_names:
                        continue
                    loaded_deferred_names.add(name)
                    deferred_list.append(tool)
                if deferred_list:
                    names = [tool.name for tool in deferred_list]
                    tool_call_key = str(msg.get("tool_call_id"))
                    names_key = ",".join(names)
                    search_call_id = f"pi_tool_load_{short_hash(f'{tool_call_key}:{names_key}')}"
                    items.append(
                        {
                            "type": "tool_search_call",
                            "call_id": search_call_id,
                            "execution": "client",
                            "status": "completed",
                            "arguments": {"query": " ".join(names), "limit": len(names)},
                        }
                    )
                    items.append(
                        {
                            "type": "tool_search_output",
                            "call_id": search_call_id,
                            "execution": "client",
                            "status": "completed",
                            "tools": to_responses_tools(
                                deferred_list,
                                supports_strict_mode=(
                                    supports_strict_mode(model) if model is not None else True
                                ),
                                defer_loading=True,
                            ),
                        }
                    )

        msg_index += 1

    return items


async def responses_stream(
    model: Model,
    context: Context,
    api_key: str,
    base_url: str = "",
    options: StreamOptions | None = None,
    *,
    client_factory: Callable[..., AsyncOpenAI] | None = None,
    request_model_id: str | None = None,
) -> AssistantMessageEventStream:
    """
    执行一次 Responses API

    流式请求。

    返回：

    AssistantMessageEventStream。

    该函数立即返回 EventStream，

    真正的网络请求在后台协程中执行。

    调用者可以立刻：

        async for

    监听模型输出。
    """

    # SDK 内部事件流。
    #
    # Responses Event
    #
    # 会转换成这里的 Event。
    stream = AssistantMessageEventStream()
    opts = options or {}

    async def _run() -> None:
        """
        后台协程。

        负责：

        创建 Client

        ↓

        发送请求

        ↓

        读取 Responses Event

        ↓

        转换 SDK Event

        ↓

        生成最终 AssistantMessage
        """

        client: Any = None
        try:
            # 创建 OpenAI SDK 客户端。
            #
            # 重试参数从 StreamOptions 读取（缺省保持 SDK 默认 2）。
            # retry-after / retry-after-ms / x-should-retry / 408/409/429/5xx
            # 由 openai SDK 内置处理，指数退避封顶 60s。
            # （max_retry_delay_ms 暂不生效：SDK 客户端不接受该参数。）
            timeout_ms = opts.get("timeout_ms")
            request_headers = dict(opts.get("headers") or {})
            if model.provider == "github-copilot":
                # 动态头（对齐 TS buildCopilotDynamicHeaders）。
                request_headers.update(
                    build_copilot_dynamic_headers(
                        messages=context.messages,
                        has_images=has_copilot_vision_input(context.messages),
                    )
                )
            resolved_headers = request_headers or None
            if client_factory is not None:
                client = client_factory(
                    api_key,
                    base_url,
                    timeout=timeout_ms / 1000.0 if timeout_ms else 180.0,
                    max_retries=opts.get("max_retries", 2),
                    headers=resolved_headers,
                )
            else:
                client = _create_client(
                    api_key,
                    base_url,
                    timeout=timeout_ms / 1000.0 if timeout_ms else 180.0,
                    max_retries=opts.get("max_retries", 2),
                    headers=resolved_headers,
                )
            # 请求参数构造（与 Codex WebSocket 路径共享，保证 body 一致）。
            kwargs, web_search_enabled = _build_responses_request_kwargs(
                model,
                context,
                base_url,
                options,
                request_model_id=request_model_id,
            )

            # 发起流式请求。
            #
            # 返回异步 Event Stream。
            response = await client.responses.create(**kwargs)

            # 最终 AssistantMessage.content。
            content_blocks: list[ContentBlock] = []

            # 当前正在累积的内容块（text / thinking / toolCall）。
            current_index: int | None = None
            current_kind: str | None = None

            # 并行工具调用状态：双段 ID → {block_index, raw_arguments, name}。
            #
            # function_call_arguments.delta 无归属信息（协议按 item 串行下发），
            # 追加到最后一个 added 的调用；done 按 added 顺序（FIFO）结束，
            # 避免 done 事件错误结束"当前"块导致前一个调用参数提前清空。
            tool_call_states: dict[str, dict[str, Any]] = {}
            pending_tool_call_ids: list[str] = []
            last_tool_call_id: str | None = None

            # Token 使用统计。
            usage: Usage = empty_usage()
            stop_reason: StopReason = "stop"
            responses_items: list[dict[str, Any]] = []
            reasoning_blocks_by_id: dict[str, dict[str, Any]] = {}
            failed = False

            def _partial() -> AssistantMessage:
                """构造当前累积状态的 partial AssistantMessage 快照。"""
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
                """结束当前累积块并发射对应的 *_end 事件。"""
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
                            content=thinking_block["thinking"],
                            partial=_partial(),
                        )
                    )
                current_kind = None
                current_index = None

            def _end_tool_call(state_id: str) -> None:
                """结束指定的工具调用块并发射 toolcall_end（幂等）。"""
                nonlocal current_kind, current_index, last_tool_call_id
                state = tool_call_states.pop(state_id, None)
                if state is None:
                    return
                if state_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(state_id)
                tool_block = cast(ToolCall, content_blocks[state["block_index"]])
                tool_block["raw_arguments"] = state["raw_arguments"]
                tool_block["arguments"] = parse_tool_arguments(state["raw_arguments"])
                stream.push(
                    ToolCallEndEvent(
                        type="toolcall_end",
                        content_index=state["block_index"],
                        tool_call=tool_block,
                        partial=_partial(),
                    )
                )
                if current_kind == "toolCall" and current_index == state["block_index"]:
                    current_kind = None
                    current_index = None
                    last_tool_call_id = None

            def _backfill_reasoning_signatures(response: Any) -> None:
                """终态 response 补回 reasoning.encrypted_content（对齐 TS）。"""
                output = getattr(response, "output", None) or []
                for item in output:
                    if getattr(item, "type", None) != "reasoning":
                        continue
                    encrypted = getattr(item, "encrypted_content", None)
                    item_id = getattr(item, "id", None)
                    if not encrypted or not isinstance(item_id, str):
                        continue
                    block = reasoning_blocks_by_id.get(item_id)
                    if block is None or not block.get("thinking_signature"):
                        continue
                    try:
                        stored = json.loads(block["thinking_signature"])
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(stored, dict) or stored.get("encrypted_content"):
                        continue
                    stored["encrypted_content"] = encrypted
                    block["thinking_signature"] = json.dumps(stored, ensure_ascii=False)

            def _begin_text() -> int:
                """确保当前块为文本块，返回 content_index。"""
                nonlocal current_kind, current_index
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
                return current_index  # type: ignore[return-value]

            def _begin_thinking() -> int:
                """确保当前块为思考块，返回 content_index。"""
                nonlocal current_kind, current_index
                if current_kind != "thinking":
                    _end_current_block()
                    current_kind = "thinking"
                    content_blocks.append(ThinkingContent(type="thinking", thinking=""))
                    current_index = len(content_blocks) - 1
                    stream.push(
                        ThinkingStartEvent(
                            type="thinking_start",
                            content_index=current_index,
                            partial=_partial(),
                        )
                    )
                return current_index  # type: ignore[return-value]

            # 流开始事件。
            stream.push(StartEvent(type="start", partial=_partial()))

            # 持续读取 Responses Event。
            #
            # 不同事件
            #
            # 分别处理。
            async for event in response:
                event_type = getattr(event, "type", None)

                # 文本增量。
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    idx = _begin_text()
                    cast(TextContent, content_blocks[idx])["text"] += delta
                    stream.push(
                        TextDeltaEvent(
                            type="text_delta",
                            content_index=idx,
                            delta=delta,
                            partial=_partial(),
                        )
                    )

                # 文本块完成（DeepSeek 可能只发 done 不发 delta）。
                elif event_type == "response.output_text.done":
                    text = getattr(event, "text", "") or ""
                    if text:
                        if current_kind == "text" and current_index is not None:
                            cast(TextContent, content_blocks[current_index])["text"] = text
                        else:
                            idx = _begin_text()
                            cast(TextContent, content_blocks[idx])["text"] = text

                # 拒绝内容（refusal）并入文本输出。
                elif event_type == "response.refusal.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        idx = _begin_text()
                        cast(TextContent, content_blocks[idx])["text"] += delta
                        stream.push(
                            TextDeltaEvent(
                                type="text_delta",
                                content_index=idx,
                                delta=delta,
                                partial=_partial(),
                            )
                        )

                # 新 Tool Call。
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", None) == "function_call":
                        # 并行 function_call 不提前结束前一个调用
                        # （其参数增量/结束事件仍可能到达）。
                        if current_kind in ("text", "thinking"):
                            _end_current_block()
                        current_kind = "toolCall"
                        call_id = getattr(item, "call_id", "") or ""
                        item_id = getattr(item, "id", None) or ""
                        # 保存完整双段 ID（call_id|fc_item_id），
                        # 供后续轮次回放时的跨 provider 规范化。
                        state_id = f"{call_id}|{item_id}" if item_id else call_id
                        last_tool_call_id = state_id
                        content_blocks.append(
                            ToolCall(
                                type="toolCall",
                                id=state_id,
                                name=getattr(item, "name", "") or "",
                                raw_arguments="",
                                arguments=None,
                            )
                        )
                        current_index = len(content_blocks) - 1
                        tool_call_states[state_id] = {
                            "block_index": current_index,
                            "raw_arguments": "",
                            "name": getattr(item, "name", "") or "",
                        }
                        pending_tool_call_ids.append(state_id)
                        stream.push(
                            ToolCallStartEvent(
                                type="toolcall_start",
                                content_index=current_index,
                                partial=_partial(),
                            )
                        )

                # Tool 参数增量。
                elif event_type == "response.function_call_arguments.delta":
                    delta = getattr(event, "delta", "")
                    if last_tool_call_id is not None and last_tool_call_id in tool_call_states:
                        state = tool_call_states[last_tool_call_id]
                        state["raw_arguments"] += delta
                        index = state["block_index"]
                        cast(ToolCall, content_blocks[index])["raw_arguments"] = state[
                            "raw_arguments"
                        ]
                    else:
                        index = current_index or 0
                    stream.push(
                        ToolCallDeltaEvent(
                            type="toolcall_delta",
                            content_index=index,
                            delta=delta,
                            partial=_partial(),
                        )
                    )

                # Tool 参数结束。
                elif event_type == "response.function_call_arguments.done":
                    # done 按 added 顺序（FIFO）匹配：串行协议下队首即当前调用。
                    if pending_tool_call_ids:
                        _end_tool_call(pending_tool_call_ids[0])
                    else:
                        _end_current_block()

                # 输出项完成。
                #
                # reasoning / message 等类型的完整 item 在此事件到达。
                elif event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None) if item else None

                    # function_call 项完成：按双段 ID 结束对应块
                    # （兼容未发 arguments.done 的端点）。
                    if item_type == "function_call":
                        call_id = getattr(item, "call_id", "") or ""
                        item_id = getattr(item, "id", None) or ""
                        state_id = f"{call_id}|{item_id}" if item_id else call_id
                        if state_id in tool_call_states:
                            _end_tool_call(state_id)

                    # 服务端 web_search 项：保留完整原始 item 供 stateless 回放。
                    if item_type == "web_search_call" and web_search_enabled:
                        raw_item = _to_jsonable(item)
                        if isinstance(raw_item, dict) and raw_item.get("id"):
                            responses_items.append(raw_item)

                    # Reasoning 项：
                    #
                    # 捕获完整 item（含 id / summary / content /
                    # encrypted_content）存入 thinking_signature，
                    # 供后续轮次回放（对齐 TS openai-responses-shared）。
                    if (
                        item_type == "reasoning"
                        and current_kind == "thinking"
                        and current_index is not None
                    ):
                        block = cast(dict[str, Any], content_blocks[current_index])
                        summary = getattr(item, "summary", None) or []
                        content = getattr(item, "content", None) or []
                        summary_text = "\n\n".join(getattr(p, "text", "") or "" for p in summary)
                        content_text = "\n\n".join(getattr(p, "text", "") or "" for p in content)
                        # DeepSeek 流式下 reasoning_text 通过 delta 下发，
                        # output_item.done 的 item.content 可能为空；用已累积
                        # 的 delta 文本补全回放项，否则下一轮会报
                        # "reasoning_text ... must be passed back"。
                        delta_text = block["thinking"]
                        block["thinking"] = summary_text or content_text or delta_text
                        raw_item = _to_jsonable(item)
                        if (
                            isinstance(raw_item, dict)
                            and not raw_item.get("content")
                            and not raw_item.get("encrypted_content")
                            and delta_text
                        ):
                            raw_item["content"] = [
                                {
                                    "type": "reasoning_text",
                                    "text": delta_text,
                                    "annotations": [],
                                }
                            ]
                        block["thinking_signature"] = json.dumps(raw_item, ensure_ascii=False)
                        item_id = raw_item.get("id") if isinstance(raw_item, dict) else None
                        if isinstance(item_id, str) and item_id:
                            reasoning_blocks_by_id[item_id] = block
                        stream.push(
                            ThinkingEndEvent(
                                type="thinking_end",
                                content_index=current_index,
                                content=block["thinking"],
                                partial=_partial(),
                            )
                        )
                        current_kind = None
                        current_index = None

                    # 消息项：持久化 text_signature 供后续轮次回放。
                    if (
                        item_type == "message"
                        and current_kind == "text"
                        and current_index is not None
                    ):
                        item_id = getattr(item, "id", None)
                        if isinstance(item_id, str) and item_id:
                            phase = getattr(item, "phase", None)
                            text_block = cast(TextContent, content_blocks[current_index])
                            text_block["text_signature"] = encode_text_signature_v1(
                                item_id, phase if isinstance(phase, str) else None
                            )

                # 模型推理内容。
                elif event_type == "response.reasoning_summary_part.added":
                    summary = getattr(event, "part", None)
                    if summary and getattr(summary, "type", None) == "summary_text":
                        text = getattr(summary, "text", "")
                        idx = _begin_thinking()
                        cast(ThinkingContent, content_blocks[idx])["thinking"] += text
                        stream.push(
                            ThinkingDeltaEvent(
                                type="thinking_delta",
                                content_index=idx,
                                delta=text,
                                partial=_partial(),
                            )
                        )

                elif event_type == "response.reasoning_text.delta":
                    delta = getattr(event, "delta", "")
                    idx = _begin_thinking()
                    cast(ThinkingContent, content_blocks[idx])["thinking"] += delta
                    stream.push(
                        ThinkingDeltaEvent(
                            type="thinking_delta",
                            content_index=idx,
                            delta=delta,
                            partial=_partial(),
                        )
                    )

                elif event_type == "response.reasoning_text.done":
                    text = getattr(event, "text", "") or ""
                    if text:
                        if current_kind == "thinking" and current_index is not None:
                            cast(ThinkingContent, content_blocks[current_index])["thinking"] = text
                        else:
                            idx = _begin_thinking()
                            cast(ThinkingContent, content_blocks[idx])["thinking"] = text

                elif event_type == "response.reasoning_summary_text.delta":
                    delta = getattr(event, "delta", "")
                    idx = _begin_thinking()
                    cast(ThinkingContent, content_blocks[idx])["thinking"] += delta
                    stream.push(
                        ThinkingDeltaEvent(
                            type="thinking_delta",
                            content_index=idx,
                            delta=delta,
                            partial=_partial(),
                        )
                    )

                elif event_type == "response.reasoning_summary_text.done":
                    text = getattr(event, "text", "") or ""
                    if text:
                        if current_kind == "thinking" and current_index is not None:
                            cast(ThinkingContent, content_blocks[current_index])["thinking"] = text
                        else:
                            idx = _begin_thinking()
                            cast(ThinkingContent, content_blocks[idx])["thinking"] = text

                # 整个 Responses 请求结束。
                elif event_type == "response.completed":
                    resp = getattr(event, "response", None)
                    if resp:
                        _backfill_reasoning_signatures(resp)
                        # 权威输出文本：覆盖实时累积的 text 块；
                        # 若尚未有 text 块（如仅 completed 事件），则新建。
                        output_text = getattr(resp, "output_text", "")
                        if output_text:
                            text_blocks = [
                                block for block in content_blocks if block.get("type") == "text"
                            ]
                            if len(text_blocks) == 1:
                                # output_text 是整段文本的权威值；已有 text 块时
                                # 直接覆盖，避免 toolCall 结束后再次追加，导致回放时
                                # function_call 与 function_call_output 之间插入
                                # 多余 message item（DeepSeek 会报
                                # "No tool output found for tool call ..."）。
                                cast(TextContent, text_blocks[0])["text"] = output_text
                            elif not text_blocks:
                                _end_current_block()
                                current_kind = "text"
                                content_blocks.append(TextContent(type="text", text=output_text))
                                current_index = len(content_blocks) - 1
                                stream.push(
                                    TextStartEvent(
                                        type="text_start",
                                        content_index=current_index,
                                        partial=_partial(),
                                    )
                                )
                                # 多个 text 块（文本→toolCall→文本）：各块已按
                                # delta/done 累积自身文本，整体覆盖第一个块会把
                                # 整段 output_text 重复进最终消息，跳过覆盖。
                                stream.push(
                                    TextEndEvent(
                                        type="text_end",
                                        content_index=current_index,
                                        content=output_text,
                                        partial=_partial(),
                                    )
                                )

                        usage = _parse_response_usage(resp, model)

                # 请求提前结束（max_output_tokens / content_filter 等）。
                elif event_type == "response.incomplete":
                    resp = getattr(event, "response", None)
                    if resp:
                        _backfill_reasoning_signatures(resp)
                        usage = _parse_response_usage(resp, model)
                        incomplete = getattr(resp, "incomplete_details", None)
                        reason = getattr(incomplete, "reason", "") or ""
                        stop_reason = "length" if reason == "max_output_tokens" else "error"

                # 请求失败：推送 error 事件，不再补 done。
                elif event_type == "response.failed":
                    failed = True
                    resp = getattr(event, "response", None)
                    response_error = getattr(resp, "error", None) if resp else None
                    message = getattr(response_error, "message", "") or "response failed"
                    err_msg = build_error_message(model, RuntimeError(message))
                    stream.push({"type": "error", "reason": "error", "error": err_msg})

            # 所有事件已处理完成，
            #
            # 结束最后一个块（若有）。
            _end_current_block()
            # 残留未收到 done 事件的工具调用（兼容端点）统一收尾。
            for state_id in list(pending_tool_call_ids):
                _end_tool_call(state_id)

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
            if responses_items:
                msg["responses_items"] = responses_items
            if not failed:
                stream.push(
                    {
                        "type": "done",
                        "reason": cast(Any, stop_reason),
                        "message": msg,
                    }
                )
            # stream.end(msg)

        except asyncio.CancelledError:
            # 让 await stream.result() 抛出取消异常，而不是永久挂起。
            stream.error(asyncio.CancelledError())
            raise

        except Exception as exc:
            err_msg = build_error_message(model, exc)
            stream.push({"type": "error", "reason": "error", "error": err_msg})
            # stream.end(err_msg)
        finally:
            # 显式关闭客户端：openai SDK 依赖 __del__ 调度异步关闭，
            # 取消/异常路径与循环引用场景下可能永不执行（连接池泄漏）。
            await close_async_client(client)

    track_background_task(_run())
    return stream
