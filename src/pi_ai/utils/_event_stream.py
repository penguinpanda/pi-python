"""
异步事件流（EventStream）

这是一个基于 asyncio.Queue 实现的通用异步事件流。

它的作用类似于：

    生产者(Producer)
            │
            │ push(event)
            ▼
    asyncio.Queue（事件缓冲区）
            │
            │ async for
            ▼
    消费者(Consumer)

生产者可以不断 push() 新事件，
消费者则可以通过

    async for event in stream:

持续读取事件，而不用关心底层的同步问题。

除此之外，EventStream 还支持：

1. 等待最终结果（result()）
2. 在收到结束事件时自动结束

结束事件（例如 done、error）的定义由具体子类决定，
EventStream 本身并不关心事件的具体含义。

因此，它既可以作为"事件流"，
又可以通过 result() 获取最终结果。
"""

import asyncio

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    TypeVar,
    cast,
)

# ------------------------------------------------------
# 泛型定义
# ------------------------------------------------------

# T：事件(Event)类型
T = TypeVar("T")

# R：最终返回(Result)类型
R = TypeVar("R")

# 用于通知消费者："事件流已经结束"
_SENTINEL = object()


class EventStream(Generic[T, R]):
    """
    通用异步事件流。

    泛型参数：

        T:
            事件类型
            例如：
                str
                dict
                AssistantMessageEvent

        R:
            最终返回值类型
            例如：
                str
                AssistantMessage

    EventStream 同时提供两种能力：

    ① 持续产生事件(async for)

    ② 当收到结束事件时，提取最终结果(await result())

    其中：

    - 什么事件表示结束，由 is_complete() 决定；
    - 最终结果是什么，由 extract_result() 决定。

    因此 EventStream 本身不依赖任何具体事件类型，
    可以复用于不同的流式协议。

    两者互不影响。
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ):
        """
        Parameters
        ----------
        is_complete
            判断某个事件是否意味着整个事件流结束。

            例如：

                lambda event: event["type"] == "done"

        extract_result
            当事件流结束时，
            从最后一个事件中提取最终结果。

            例如：

                lambda event: event["message"]
        """

        # --------------------------------------------------
        # asyncio.Queue
        #
        # 所有 push() 进入的事件都会先进入队列，
        # async for 会不断从这里读取。
        # --------------------------------------------------
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

        # 是否已经结束
        self._done = False

        # --------------------------------------------------
        # Future 用来保存最终结果
        #
        # await stream.result()
        #
        # 本质上就是 await 这个 Future。
        # --------------------------------------------------
        self._result: asyncio.Future[R] = (
            asyncio.get_running_loop().create_future()
        )

        # 判断结束事件
        self._is_complete = is_complete

        # 提取最终结果
        self._extract_result = extract_result

    # ------------------------------------------------------
    # Producer API
    # ------------------------------------------------------

    def push(self, event: T) -> None:
        """
        推送一个新的事件。

        Producer 会不断调用：

            stream.push(event)

        Consumer 则通过：

            async for event in stream

        读取这些事件。
        """

        # 如果已经结束，则忽略后续事件
        if self._done:
            return

        # --------------------------------------------------
        # 判断当前事件是不是"结束事件"
        #
        # EventStream 不关心具体是哪一种事件，
        # 是否结束完全由 is_complete() 决定。
        #
        # 例如：
        #
        # done
        # error
        # finish
        #
        # 都可以作为结束事件。
        #
        # 例如：
        #
        # {
        #     "type": "done",
        #     "message": ...
        # }
        # --------------------------------------------------
        if self._is_complete(event):
            self._done = True

            # Future 只允许设置一次结果
            if not self._result.done():
                self._result.set_result(
                    self._extract_result(event)
                )

        # 不管是不是结束事件，
        # 都要放入队列，
        # 这样消费者还能收到最后一个事件。
        self._queue.put_nowait(event)

    def end(self, result: R | None = None) -> None:
        """
        手动结束事件流。

        与 push(done_event) 不同的是：

        不需要再发送一个事件，
        可以直接结束整个 EventStream。

        Parameters
        ----------
        result

            可选的最终返回值。

            如果提供：

                await stream.result()

            将返回它。
        """

        self._done = True

        # 无论是否提供结果，都完成 Future。
        #
        # 否则当调用方在 `async for` 之外执行：
        #
        #     await stream.result()
        #
        # 而流又通过 end() 结束（例如任务被取消）时，
        # Future 永远无法完成，导致 result() 永久挂起。
        if not self._result.done():
            self._result.set_result(result)  # type: ignore[arg-type]

        # 放入一个特殊对象通知 async for 停止
        self._queue.put_nowait(_SENTINEL)

    def error(self, exc: BaseException) -> None:
        """
        使用异常结束整个事件流。

        之后：

            await stream.result()

        会直接抛出该异常。

        Parameters
        ----------
        exc
            结束事件流的异常。

            使用 BaseException 以兼容：

            - Exception（普通错误）
            - asyncio.CancelledError（任务取消，继承自 BaseException）
        """

        self._done = True

        if not self._result.done():
            self._result.set_exception(exc)

        # 同样通知 async for 停止
        self._queue.put_nowait(_SENTINEL)

    # ------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------

    async def result(self) -> R:
        """
        等待事件流结束，并返回最终结果。

        最终结果由 extract_result() 从结束事件中提取，
        因此不同类型的 EventStream 可以返回不同的数据。

        与遍历事件不同：

            async for

        会不断收到中间事件。

        而：

            await result()

        只关心最终结果。
        """

        return await self._result

    # ------------------------------------------------------
    # async for 支持
    # ------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[T]:
        """
        让对象支持：

            async for event in stream:

        工作流程：

                    push(event)
                         │
                         ▼
                 asyncio.Queue
                         │
                         ▼
            __aiter__() 不断 get()
                         │
                         ▼
                    yield event
        """

        while True:

            # --------------------------------------------------
            # 优先读取已经缓存好的事件
            #
            # get_nowait() 不会阻塞。
            # --------------------------------------------------
            try:
                event = self._queue.get_nowait()

            except asyncio.QueueEmpty:

                # 队列为空

                # 如果事件流已经结束，
                # 则直接退出 async for。
                if self._done:
                    return

                # --------------------------------------------------
                # 否则等待新的事件。
                #
                # await get() 会挂起协程，
                # 直到 Producer push 新事件。
                # --------------------------------------------------
                event = await self._queue.get()

            # --------------------------------------------------
            # Sentinel 表示整个事件流结束
            # --------------------------------------------------
            if event is _SENTINEL:
                return

            # 返回一个事件
            yield event


# ==========================================================
# AssistantMessage 专用事件流
# ==========================================================

from .._types import AssistantMessage, AssistantMessageEvent, ErrorEvent


class AssistantMessageEventStream(
    EventStream[AssistantMessageEvent, AssistantMessage]
):
    """
    AssistantMessage 专用事件流。

    泛型具体化后：

        T = AssistantMessageEvent

        R = AssistantMessage

    即：

        async for
            得到的是 AssistantMessageEvent

        await result()
            得到的是 AssistantMessage
    """

    def __init__(self) -> None:
        super().__init__(
            
            # --------------------------------------------------
            # 收到 done 或 error 即认为整个流结束
            # --------------------------------------------------
            is_complete=lambda e: e["type"] in ("done", "error"),

            # --------------------------------------------------
            # done:
            #
            # {
            #     "type":"done",
            #     "message": AssistantMessage
            # }
            #
            # 返回 message
            #
            # error:
            #
            # {
            #     "type": "error",
            #     "reason": "...",
            #     "error": AssistantMessage
            # }
            #
            # 返回 error 中携带的 AssistantMessage。
            # --------------------------------------------------
            extract_result=lambda e: (
                e["message"]
                if e["type"] == "done"
                else cast(ErrorEvent, e)["error"]
            ),
        )