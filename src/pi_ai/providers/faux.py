"""
Faux Provider（测试用假 Provider）

=========================================================
模块职责
=========================================================

本模块提供一个完全进程内、零网络依赖的模型 Provider，
用于测试 agent 层代码。

它不访问任何真实 API，而是：

    ① 预置（脚本化）响应序列

    ② 按调用顺序逐个返回

    ③ 支持基于上下文的动态响应工厂

    ④ 模拟流式输出（text_delta / toolcall_delta / thinking_delta）

    ⑤ 估算 Token Usage

=========================================================
使用示例
=========================================================

    from pi_ai import Context
    from pi_ai.providers.faux import faux_provider, faux_assistant_message

    faux = faux_provider()
    models = Models()
    models.add_provider(faux.provider)
    faux.set_responses([
        faux_assistant_message("Hello, how can I help?"),
    ])

    model = faux.get_model()
    result = await models.complete(model, Context(messages=[...]))
"""

import asyncio
import json
import math
import time
from typing import Any, Awaitable, Callable, Sequence, cast

from .._event_stream import AssistantMessageEventStream
from .._types import (
    AssistantMessage,
    ContentBlock,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ModelCost,
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
from ..api._shared import empty_usage
from ..provider import Provider, create_provider

# 默认值
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_TOKEN_SIZE = 4  # 每个流式 chunk 的 token 数（字符数 = token * 4）


# ------------------------------------------------------
# Faux 默认模型。
#
# 注意：Model dataclass 没有 contextWindow 字段。
# ------------------------------------------------------
FAUX_MODEL = Model(
    id=DEFAULT_MODEL_ID,
    provider=DEFAULT_PROVIDER,
    api="openai-completions",
    name=DEFAULT_MODEL_NAME,
    input=["text"],
    output=["text"],
    maxTokens=16384,
    reasoning=False,
    supportsToolCalling=True,
    supportsImages=False,
    cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
)


# ------------------------------------------------------
# Faux 辅助函数
# ------------------------------------------------------


def faux_text(text: str) -> TextContent:
    """创建 faux 文本内容块。"""
    return TextContent(type="text", text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    """创建 faux 思考内容块。"""
    return ThinkingContent(type="thinking", thinking=thinking)


def faux_tool_call(
    name: str,
    args: dict[str, Any] | str,
    *,
    tool_call_id: str | None = None,
) -> ToolCall:
    """创建 faux 工具调用内容块。

    args 可以是 dict（解析后的参数对象）
    或 JSON 字符串（自动解析）。
    """
    if isinstance(args, str):
        parsed: Any = json.loads(args) if args.strip() else {}
    else:
        parsed = args
    return ToolCall(
        type="toolCall",
        id=tool_call_id or f"tool:{int(time.time() * 1000)}",
        name=name,
        arguments=parsed if isinstance(parsed, dict) else {"value": parsed},
    )


def _normalize_content(
    content: str | ContentBlock | list[ContentBlock],
) -> list[ContentBlock]:
    """将 content 统一为 ContentBlock 列表。"""
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return content
    return [content]


def faux_assistant_message(
    content: str | ContentBlock | list[ContentBlock],
    *,
    stop_reason: StopReason = "stop",
    error_message: str | None = None,
    usage: Usage | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    provider_id: str = DEFAULT_PROVIDER,
) -> AssistantMessage:
    """创建脚本化的 AssistantMessage。

    用于设置 Faux Provider 的响应序列。

    content 支持：

        - str：自动包装为单个 TextContent
        - 单个 ContentBlock
        - ContentBlock 列表
    """
    return AssistantMessage(
        role="assistant",
        content=_normalize_content(content),
        api="openai-completions",
        provider=provider_id,
        model=model_id,
        usage=usage or empty_usage(),
        stopReason=stop_reason,
        errorMessage=error_message,
        timestamp=now_ms(),
    )


# ------------------------------------------------------
# Faux 响应类型
# ------------------------------------------------------

# 动态响应工厂。
#
# 接收：
#
#     context   本次请求的完整上下文
#     options   流式请求参数
#     state     共享状态（含 callCount）
#     model     本次请求使用的模型
#
# 返回 AssistantMessage。
FauxResponseFactory = Callable[
    [Context, StreamOptions | None, dict[str, Any], Model],
    Awaitable[AssistantMessage],
]

FauxResponseStep = AssistantMessage | FauxResponseFactory


# ------------------------------------------------------
# Token 估算
# ------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：每 4 个字符约 1 个 token。"""
    return math.ceil(len(text) / 4)


def _blocks_to_text(blocks: Sequence[ContentBlock]) -> str:
    """将 ContentBlock 列表转换为纯文本（用于 token 估算）。

    使用直接比较（而非中间变量）让类型检查器正确收窄联合类型。
    """
    parts: list[str] = []
    for block in blocks:
        if block["type"] == "text":
            parts.append(block["text"])
        elif block["type"] == "image":
            media_type = block.get("mimeType")
            data_len = len(block.get("data") or "")
            parts.append(f"[image:{media_type}:{data_len}]")
        elif block["type"] == "thinking":
            parts.append(block["thinking"])
        elif block["type"] == "toolCall":
            parts.append(f"{block['name']}:{json.dumps(block['arguments'], ensure_ascii=False)}")
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    """将单条 Message 转换为纯文本（用于 token 估算）。"""
    if message["role"] == "user":
        content = message["content"]
        return content if isinstance(content, str) else _blocks_to_text(content)
    if message["role"] == "assistant":
        return _blocks_to_text(message["content"])
    if message["role"] == "toolResult":
        return f"{message['toolName']}\n{_blocks_to_text(message['content'])}"
    # system：SystemMessage 没有参与 token 估算的内容。
    return ""


def _tool_to_dict(tool: Tool) -> dict[str, Any]:
    """将 Tool dataclass 转换为可 JSON 序列化的 dict。"""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
    }


def _serialize_context(context: Context) -> str:
    """将 Context 序列化为纯文本，用于估算输入 token。"""
    parts: list[str] = []
    if context.systemPrompt:
        parts.append(f"system:{context.systemPrompt}")
    for message in context.messages:
        parts.append(f"{message['role']}:{_message_to_text(message)}")
    if context.tools:
        parts.append(
            f"tools:{json.dumps([_tool_to_dict(t) for t in context.tools], ensure_ascii=False)}"
        )
    return "\n\n".join(parts)


def _assistant_content_to_text(content: list[ContentBlock]) -> str:
    return _blocks_to_text(content)


# ------------------------------------------------------
# Faux Core
# ------------------------------------------------------


class FauxCore:
    """Faux Provider 的核心状态与流实现。"""

    # 由 faux_provider() 填充的 Provider 实例。
    provider: Provider

    def __init__(
        self,
        models: list[Model],
        *,
        tokens_per_second: int = 0,
        token_size: int = DEFAULT_TOKEN_SIZE,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        # 脚本化响应队列。
        self._responses: list[FauxResponseStep] = []

        # 下一个待消费响应的下标。
        self._response_index = 0

        # 共享状态，供响应工厂读取。
        self._state: dict[str, Any] = {"callCount": 0}

        self._tokens_per_second = tokens_per_second
        self._token_size = token_size
        self._provider = provider
        self._models: list[Model] = models or [FAUX_MODEL]

    # ------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------

    @property
    def models(self) -> list[Model]:
        """Faux Provider 暴露的模型列表。"""
        return self._models

    @property
    def state(self) -> dict[str, Any]:
        """共享状态（含 callCount）。"""
        return self._state

    @property
    def call_count(self) -> int:
        """已消费的响应总数。"""
        return int(self._state.get("callCount", 0))

    # ------------------------------------------------------
    # 响应控制
    # ------------------------------------------------------

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        """设置响应序列（覆盖已有）。"""
        self._responses = list(responses)
        self._response_index = 0

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        """追加响应到序列末尾。"""
        self._responses.extend(responses)

    def get_pending_response_count(self) -> int:
        """获取剩余待消费的响应数。"""
        return max(0, len(self._responses) - self._response_index)

    def get_model(self, model_id: str | None = None) -> Model | None:
        """按 ID 查询模型；不传返回第一个。"""
        if model_id is None:
            return self._models[0]
        for candidate in self._models:
            if candidate.id == model_id:
                return candidate
        return None

    # ------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------

    async def _get_next_response(
        self,
        context: Context,
        options: StreamOptions | None,
        model: Model,
    ) -> AssistantMessage:
        """获取下一个响应（支持静态消息与动态工厂）。"""
        # 每次调用都递增 callCount（包括耗尽时），
        # 与 TS 行为一致：callCount 表示模型调用次数。
        self._state["callCount"] = self.call_count + 1

        if self._response_index >= len(self._responses):
            return faux_assistant_message(
                [],
                stop_reason="error",
                error_message="No more faux responses queued",
                model_id=model.id,
                provider_id=model.provider,
            )

        step = self._responses[self._response_index]
        self._response_index += 1

        if callable(step):
            return await step(context, options, self._state, model)
        return step

    def _rewrite_message(
        self,
        message: AssistantMessage,
        model: Model,
    ) -> AssistantMessage:
        """将消息的 api/provider/model 重写为请求模型。"""
        return cast(AssistantMessage, {
            **message,
            "api": model.api,
            "provider": model.provider,
            "model": model.id,
        })

    def _with_usage(
        self,
        message: AssistantMessage,
        context: Context,
    ) -> AssistantMessage:
        """基于序列化上下文与输出内容估算 Usage。"""
        prompt_tokens = _estimate_tokens(_serialize_context(context))
        output_tokens = _estimate_tokens(_assistant_content_to_text(message["content"]))
        return cast(AssistantMessage, {
            **message,
            "usage": Usage(
                input=prompt_tokens,
                output=output_tokens,
                cacheRead=0,
                cacheWrite=0,
                totalTokens=prompt_tokens + output_tokens,
                cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            ),
        })

    def _chunk_text(self, text: str) -> list[str]:
        """按 token_size 将文本拆分为流式 chunk。"""
        char_size = max(1, self._token_size * 4)
        return [text[i : i + char_size] for i in range(0, len(text), char_size)]

    async def _delay(self, text: str) -> None:
        """模拟流式速度（tokens_per_second）。0 表示无延迟。"""
        if self._tokens_per_second <= 0:
            return
        await asyncio.sleep(_estimate_tokens(text) / self._tokens_per_second)

    def _push_aborted(self, stream: AssistantMessageEventStream, message: AssistantMessage) -> None:
        """以 aborted 结束流。"""
        aborted = cast(AssistantMessage, {
            **message,
            "stopReason": "aborted",
            "errorMessage": "Request was aborted",
        })
        stream.push(ErrorEvent(type="error", reason="aborted", error=aborted))
        stream.end(aborted)

    async def _stream_response(
        self,
        stream: AssistantMessageEventStream,
        message: AssistantMessage,
        opts: StreamOptions | None,
    ) -> None:
        """将脚本化消息拆分为增量事件输出。"""
        stop_reason = message.get("stopReason", "stop")

        if stop_reason == "pending":
            raise RuntimeError("Faux response ended without a stop reason")

        # 可选的中止信号（asyncio.Event）。
        signal = (opts or {}).get("signal")

        # partial 快照：faux 直接使用完整脚本化消息。
        def _partial() -> AssistantMessage:
            return message

        for content_index, block in enumerate(message["content"]):
            if block["type"] == "text":
                stream.push(TextStartEvent(
                    type="text_start", contentIndex=content_index, partial=_partial(),
                ))
                for chunk in self._chunk_text(block["text"]):
                    if signal and signal.is_set():
                        self._push_aborted(stream, message)
                        return
                    stream.push(TextDeltaEvent(
                        type="text_delta", contentIndex=content_index, delta=chunk, partial=_partial(),
                    ))
                    await self._delay(chunk)
                stream.push(TextEndEvent(
                    type="text_end", contentIndex=content_index, content=block["text"], partial=_partial(),
                ))
            elif block["type"] == "thinking":
                stream.push(ThinkingStartEvent(
                    type="thinking_start", contentIndex=content_index, partial=_partial(),
                ))
                for chunk in self._chunk_text(block["thinking"]):
                    if signal and signal.is_set():
                        self._push_aborted(stream, message)
                        return
                    stream.push(ThinkingDeltaEvent(
                        type="thinking_delta", contentIndex=content_index, delta=chunk, partial=_partial(),
                    ))
                    await self._delay(chunk)
                stream.push(ThinkingEndEvent(
                    type="thinking_end", contentIndex=content_index, content=block["thinking"], partial=_partial(),
                ))
            elif block["type"] == "toolCall":
                stream.push(ToolCallStartEvent(
                    type="toolcall_start", contentIndex=content_index, partial=_partial(),
                ))
                args_json = json.dumps(block["arguments"], ensure_ascii=False)
                for chunk in self._chunk_text(args_json):
                    if signal and signal.is_set():
                        self._push_aborted(stream, message)
                        return
                    stream.push(ToolCallDeltaEvent(
                        type="toolcall_delta", contentIndex=content_index, delta=chunk, partial=_partial(),
                    ))
                    await self._delay(chunk)
                stream.push(ToolCallEndEvent(
                    type="toolcall_end", contentIndex=content_index, toolCall=cast(ToolCall, block), partial=_partial(),
                ))

        if stop_reason in ("error", "aborted"):
            stream.push(ErrorEvent(type="error", reason=stop_reason, error=message))
            stream.end(message)
            return

        # 走到这里时 stop_reason 已排除 pending / error / aborted，
        # 即限定为 stop / length / toolUse。
        stream.push(DoneEvent(type="done", reason=cast(Any, stop_reason), message=message))
        stream.end(message)

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """Faux Provider 的流式实现。

        创建流并后台调度 _run()，立即返回。
        """
        stream = AssistantMessageEventStream()
        opts = options or {}

        async def _run() -> None:
            try:
                message = await self._get_next_response(context, opts, model)
                message = self._rewrite_message(message, model)
                message = self._with_usage(message, context)
                await self._stream_response(stream, message, opts)
            except asyncio.CancelledError:
                # 让 await stream.result() 抛出取消异常，而不是永久挂起。
                stream.error(asyncio.CancelledError())
                raise
            except Exception as exc:
                error_msg = faux_assistant_message(
                    [],
                    stop_reason="error",
                    error_message=str(exc),
                    model_id=model.id,
                    provider_id=model.provider,
                )
                stream.push(ErrorEvent(type="error", reason="error", error=error_msg))
                stream.end(error_msg)

        asyncio.create_task(_run())
        return stream


# ------------------------------------------------------
# 公开 API
# ------------------------------------------------------


def faux_provider(
    models: list[Model] | None = None,
    *,
    tokens_per_second: int = 0,
    token_size: int = DEFAULT_TOKEN_SIZE,
    provider: str = DEFAULT_PROVIDER,
) -> FauxCore:
    """创建 Faux Provider。

    Parameters
    ----------
    models
        要暴露的模型列表。默认使用单个 faux-1 模型。

    tokens_per_second
        模拟的 tokens/second 速度。0 表示无延迟。

    token_size
        每个流式 chunk 的 token 数（默认 4）。

    provider
        Provider ID（默认 "faux"）。

    Returns
    -------
    FauxCore
        包含 .provider 与响应控制方法。
    """
    core = FauxCore(
        models=models or [FAUX_MODEL],
        tokens_per_second=tokens_per_second,
        token_size=token_size,
        provider=provider,
    )
    core.provider = create_provider(
        id=provider,
        name="Faux",
        auth=None,
        models=core.models,
        stream_fn=core.stream,
    )
    return core
