"""懒加载机制（对齐 TS api/lazy.ts）。

- lazy_stream：同步返回 EventStream，后台执行异步 setup（认证解析、
  模块加载）；setup 失败通过 error 事件优雅降级，不抛给调用方。
- lazy_api：包装动态加载的 API 实现模块为 ProviderStreams。
- forward_stream：把 inner stream 的事件转发到 outer stream。
"""

import asyncio

from typing import AsyncIterable, Awaitable, Callable

from .._types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Model,
    ProviderStreams,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    now_ms,
)
from ..utils._event_stream import AssistantMessageEventStream

# 持有后台任务引用，避免被 GC 静默取消。
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    )


def _create_setup_error_message(model: Model, error: object) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_empty_usage(),
        stop_reason="error",
        error_message=str(error),
        timestamp=now_ms(),
    )


async def forward_stream(
    target: AssistantMessageEventStream,
    source: AsyncIterable[AssistantMessageEvent],
) -> None:
    """把 source 的事件全部转发到 target，并以 source 的结果结束 target。"""
    result = None
    has_result = callable(getattr(source, "result", None))
    async for event in source:
        target.push(event)
    if has_result:
        result = await source.result()  # type: ignore[union-attr]
    target.end(result)


def lazy_stream(
    model: Model,
    setup: Callable[[], Awaitable[AsyncIterable[AssistantMessageEvent]]],
) -> AssistantMessageEventStream:
    """同步返回流；setup（认证解析 / 模块加载）在后台执行。"""
    outer = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            inner = await setup()
            await forward_stream(outer, inner)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            message = _create_setup_error_message(model, exc)
            outer.push({"type": "error", "reason": "error", "error": message})
            outer.end(message)

    task = asyncio.create_task(_run())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return outer


def _call_api(
    name: str,
    load: Callable[[], Awaitable[ProviderStreams]],
    model: Model,
    context: Context,
    options: StreamOptions | SimpleStreamOptions | None,
) -> Callable[[], Awaitable[AsyncIterable[AssistantMessageEvent]]]:
    async def _setup() -> AsyncIterable[AssistantMessageEvent]:
        api = await load()
        fn = getattr(api, name)
        return fn(model, context, options)

    return _setup


def lazy_api(load: Callable[[], Awaitable[ProviderStreams]]) -> ProviderStreams:
    """包装动态加载的 API 实现模块为 ProviderStreams。

    模块在首次 stream / streamSimple 调用时加载；Python import 缓存
    天然去重。加载或调用失败以 error 事件结束流。
    """

    class _LazyStreams:
        def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            return lazy_stream(model, _call_api("stream", load, model, context, options))

        def streamSimple(
            self,
            model: Model,
            context: Context,
            options: SimpleStreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            return lazy_stream(
                model, _call_api("streamSimple", load, model, context, options)
            )

    return _LazyStreams()  # type: ignore[return-value]


__all__ = ["lazy_stream", "lazy_api", "forward_stream"]
