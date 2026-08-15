"""
Agent 有状态包装类

拥有运行状态、管理并发、协调生命周期的有状态包装器。

核心职责:
- 互斥运行：同一 Agent 实例同时最多一个活跃运行
- 状态管理：维护 AgentState，通过 _process_event 归约事件 → 状态
- 生命周期管理：管理 abort signal + isStreaming 标志
- 事件订阅：subscribe() 注册监听器，按订阅顺序 await，并传递当前运行的 abort signal
- 桥接：AgentOptions → AgentLoopConfig → run_agent_loop()
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pi_ai.types import (
    CacheRetention,
    ImageContent,
    Message,
    Model,
    TextContent,
    UserMessage,
    now_ms,
)
from pi_ai import RetryPolicy

from ._agent_loop import run_agent_loop, run_agent_loop_continue
from ._stream_fn import get_default_stream_fn
from ._types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareNextTurnContext,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
)

from pi_ai.types import AssistantMessage, ThinkingBudgets, Transport, Usage

# 监听器签名：async (event, signal) → None；同步监听器返回 None 也支持
AgentListener = Callable[[AgentEvent, asyncio.Event | None], Awaitable[None] | None]

DEFAULT_MODEL = Model(
    id="unknown",
    provider="unknown",
    api="unknown",
    name="Unknown model",
)

# ---------------------------------------------------------------------------
# PendingMessageQueue（双消息队列的，双重嵌套循环的输入源）
# ---------------------------------------------------------------------------


class PendingMessageQueue:
    """消息队列，支持 QueueMode 消费策略。

    - mode="all": drain() 一次取出全部消息
    - mode="one-at-a-time": drain() 每次只取出最早一条，剩余留待后续 drain
    """

    def __init__(self, mode: QueueMode = "one-at-a-time"):
        self.mode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def queued_count(self) -> int:
        """当前队列中的消息数量。"""
        return len(self._messages)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = self._messages
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


# ---------------------------------------------------------------------------
# AgentOptions
# ---------------------------------------------------------------------------


class AgentOptions:
    """Agent 构造选项。

    所有字段可选；stream_fn 可通过 set_default_stream_fn 全局注册。
    """

    def __init__(
        self,
        *,
        system_prompt: str = "You are a helpful assistant.",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
        stream_fn: StreamFn | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], list[Message]] | None = None,
        transform_context: (Callable[[list[AgentMessage]], list[AgentMessage]] | None) = None,
        get_api_key: Callable[[str], str | None] | None = None,
        before_tool_call: (
            Callable[
                [BeforeToolCallContext],
                BeforeToolCallResult | None,
            ]
            | None
        ) = None,
        after_tool_call: (
            Callable[
                [AfterToolCallContext],
                AfterToolCallResult | None,
            ]
            | None
        ) = None,
        prepare_next_turn: (Callable[[AgentContext], Any] | None) = None,
        prepare_next_turn_with_context: (Callable[[PrepareNextTurnContext], Any] | None) = None,
        should_stop_after_turn: Callable[[AgentContext], bool] | None = None,
        tool_execution: ToolExecutionMode = "parallel",
        # 消息队列消费策略（默认逐条消费）
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        # 提示缓存与会话标识（透传给 StreamOptions）
        session_id: str | None = None,
        cache_retention: CacheRetention | None = None,
        # 推理 token 预算与传输协议（透传给 StreamOptions / SimpleStreamOptions）
        thinking_budgets: ThinkingBudgets | None = None,
        transport: Transport | None = None,
        # 流选项透传（SimpleStreamOptions 子集）
        api_key: str | None = None,
        on_payload: Any = None,
        on_response: Any = None,
        max_retry_delay_ms: int | None = None,
        # 重试策略。None = 不重试（对齐 TS agent-loop）。
        retry_policy: RetryPolicy | None = None,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.thinking_level = thinking_level
        self.tools = tools or []
        self.messages = messages or []
        self.stream_fn = stream_fn
        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.get_api_key = get_api_key
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        self.should_stop_after_turn = should_stop_after_turn
        self.tool_execution = tool_execution
        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode
        self.session_id = session_id
        self.cache_retention = cache_retention
        self.thinking_budgets = thinking_budgets
        self.transport = transport
        self.api_key = api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.max_retry_delay_ms = max_retry_delay_ms
        self.retry_policy = retry_policy


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """LLM Agent —— 有状态包装器。

    用法:
        agent = Agent(options)
        agent.subscribe(lambda event: print(event))
        await agent.prompt("Hello, world!")
        await agent.wait_for_idle()
    """

    def __init__(self, options: AgentOptions | None = None):
        opts = options or AgentOptions()

        # -- 内部状态 --
        self._state = AgentState(
            system_prompt=opts.system_prompt,
            model=opts.model or DEFAULT_MODEL,
            thinking_level=opts.thinking_level,
        )
        self._state.tools = opts.tools
        self._state.messages = opts.messages

        # -- 可配置钩子（公开属性，可在构造后修改） --
        self.convert_to_llm: Callable[[list[AgentMessage]], list[Message]] = (
            opts.convert_to_llm or _default_convert_to_llm
        )
        self.transform_context: Callable[[list[AgentMessage]], list[AgentMessage]] | None = (
            _maybe_async(opts.transform_context)
        )
        self.stream_function: StreamFn | None = opts.stream_fn
        self.get_api_key: Callable[[str], str | None] | None = opts.get_api_key
        self.before_tool_call = opts.before_tool_call
        self.after_tool_call = opts.after_tool_call
        self.prepare_next_turn = opts.prepare_next_turn
        self.prepare_next_turn_with_context = opts.prepare_next_turn_with_context
        self.should_stop_after_turn = opts.should_stop_after_turn
        self.tool_execution: ToolExecutionMode = opts.tool_execution
        self.session_id: str | None = opts.session_id
        self.cache_retention: CacheRetention | None = opts.cache_retention
        self.thinking_budgets: ThinkingBudgets | None = opts.thinking_budgets
        self.transport: Transport | None = opts.transport
        self.api_key: str | None = opts.api_key
        self.on_payload: Any = opts.on_payload
        self.on_response: Any = opts.on_response
        self.max_retry_delay_ms: int | None = opts.max_retry_delay_ms
        self.retry_policy: RetryPolicy | None = opts.retry_policy

        # -- 双消息队列（steering / follow-up）--
        self._steering_queue = PendingMessageQueue(opts.steering_mode)
        self._follow_up_queue = PendingMessageQueue(opts.follow_up_mode)

        # -- 运行时 --
        self._active: bool = False
        self._abort: asyncio.Event | None = None
        self._settled = asyncio.Event()
        self._settled.set()  # 初始空闲
        self._listeners: list[AgentListener] = []

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """当前 Agent 运行时只读状态快照。"""
        return self._state

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """发送用户消息，运行完整 agent loop。阻塞直到完成。"""
        if self._active:
            raise RuntimeError(
                "Agent is already running. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )

        # 标准化输入
        prompts = _normalize_input(input, images)

        # 运行（prompts 由 run_agent_loop 注入到上下文并发出事件，
        # _process_event 处理 message_end 时会自动追加到 state.messages）
        await self._run_prompt(prompts)

    async def continue_(self) -> None:
        """从当前 transcript 继续（continue()）。

        最后一条消息为非 assistant 时直接续跑；为 assistant 时先消费队列：
        1. steering 队列非空 → 作为 prompt 运行（跳过首次 steering 轮询，
           避免与手动排空的消息重复注入）
        2. 否则 follow-up 队列非空 → 作为 prompt 运行
        3. 队列均为空 → 抛异常
        """
        if self._active:
            raise RuntimeError("Agent is already running. Wait for completion before continuing.")

        messages = self._state._messages
        if not messages:
            raise RuntimeError("No messages to continue from")

        if messages[-1].get("role") == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt(
                    queued_steering,
                    skip_initial_steering_poll=True,
                )
                return

            queued_follow_up = self._follow_up_queue.drain()
            if queued_follow_up:
                await self._run_prompt(queued_follow_up)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continue()

    def abort(self) -> None:
        """中止当前运行。"""
        if self._abort is not None:
            self._abort.set()

    @property
    def signal(self) -> asyncio.Event | None:
        """当前运行的中止信号（无运行时为 None；对齐 TS Agent.signal）。"""
        return self._abort

    def subscribe(
        self,
        listener: AgentListener,
    ) -> Callable[[], None]:
        """订阅生命周期事件。返回取消订阅函数。

        监听器按订阅顺序 await；每个监听器接收 (event, signal)，
        signal 为当前运行的取消信号（asyncio.Event，运行中非 None）。
        同步监听器（返回 None）同样支持。
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    async def wait_for_idle(self) -> None:
        """等待当前运行结束（含所有事件监听器 settle）。

        运行结束（finally）时置位 _settled；agent_end 监听器在运行体内被
        await，因此置位时所有监听器均已 settle。
        """
        if not self._settled.is_set():
            await self._settled.wait()

    # ------------------------------------------------------------------
    # 双消息队列 API（steer / follow-up）
    # ------------------------------------------------------------------

    @property
    def steering_mode(self) -> QueueMode:
        """steering 队列消费策略。"""
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """follow-up 队列消费策略。"""
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """入队一条 steering 消息：Agent 运行中途（turn 边界）注入。"""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """入队一条 follow-up 消息：Agent 即将停止时注入并继续。"""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """任一队列仍有待处理消息时返回 True。"""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    @property
    def pending_message_count(self) -> int:
        """steering + follow-up 队列中的消息总数。"""
        return self._steering_queue.queued_count() + self._follow_up_queue.queued_count()

    def reset(self) -> None:
        """清空 transcript、运行时状态和双消息队列。"""
        if self._active:
            raise RuntimeError("Cannot reset while agent is running.")
        self._state.messages = []
        self._state.streaming_message = None
        self._state.error_message = None
        self._state.pending_tool_calls = set()
        self.clear_all_queues()

    # ------------------------------------------------------------------
    # 内部：运行生命周期
    # ------------------------------------------------------------------

    async def _run_prompt(
        self,
        prompts: list[AgentMessage],
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        """执行一次完整的 prompt → agent loop 生命周期。"""
        self._active = True
        self._abort = asyncio.Event()
        self._settled.clear()
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            context = AgentContext(
                system_prompt=self._state.system_prompt,
                messages=list(self._state._messages),
                tools=list(self._state._tools),
            )
            await run_agent_loop(
                prompts=prompts,
                context=context,
                config=self._create_loop_config(
                    skip_initial_steering_poll=skip_initial_steering_poll,
                ),
                emit=self._process_event,
                signal=self._abort,
                stream_fn=self._resolve_stream_fn(),
            )
        except asyncio.CancelledError:
            # abort 触发，agent_end 事件已在 loop 中发出
            pass
        except Exception as exc:
            # 对齐 TS agent.ts runWithLifecycle：捕获所有错误并正常结束，
            # 让调用方（session）继续执行 compaction / retry 后置状态机。
            await self._emit_failure_agent_end(exc)
        finally:
            self._state.is_streaming = False
            self._active = False
            self._abort = None
            await self._process_event({"type": "agent_settled"})
            self._settled.set()

    async def _run_continue(self) -> None:
        """从当前上下文继续。"""
        self._active = True
        self._abort = asyncio.Event()
        self._settled.clear()
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            context = AgentContext(
                system_prompt=self._state.system_prompt,
                messages=list(self._state._messages),
                tools=list(self._state._tools),
            )
            await run_agent_loop_continue(
                context=context,
                config=self._create_loop_config(),
                emit=self._process_event,
                signal=self._abort,
                stream_fn=self._resolve_stream_fn(),
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # 对齐 TS runWithLifecycle：捕获所有错误并正常结束，
            # 让调用方继续执行 compaction / retry 后置状态机。
            await self._emit_failure_agent_end(exc)
        finally:
            self._state.is_streaming = False
            self._active = False
            self._abort = None
            await self._process_event({"type": "agent_settled"})
            self._settled.set()

    async def _emit_failure_agent_end(self, exc: BaseException) -> None:
        """异常路径收尾：按 TS handleRunFailure 固定四连发事件。"""
        self._state.error_message = str(exc)
        model = self._state.model
        error_msg: AssistantMessage = {
            "role": "assistant",
            "content": [TextContent(type="text", text="")],
            "api": model.api,
            "provider": model.provider,
            "model": model.id,
            "timestamp": now_ms(),
            "stop_reason": "error",
            "error_message": str(exc),
            "usage": _empty_usage(),
        }
        await self._process_event({"type": "message_start", "message": error_msg})
        await self._process_event({"type": "message_end", "message": error_msg})
        await self._process_event({"type": "turn_end", "message": error_msg, "tool_results": []})
        await self._process_event({"type": "agent_end", "messages": [error_msg]})

    # ------------------------------------------------------------------
    # 内部：桥接
    # ------------------------------------------------------------------

    def _create_loop_config(
        self,
        *,
        skip_initial_steering_poll: bool = False,
    ) -> AgentLoopConfig:
        """将 Agent 公开属性桥接为 AgentLoopConfig。

        skip_initial_steering_poll runPromptMessages 的
        skipInitialSteeringPoll：只跳过本次运行的首次 steering 轮询
        （continue() 已手动排空队列并把消息作为 prompt 注入时使用）。
        """
        skip = skip_initial_steering_poll

        async def _get_steering() -> list[AgentMessage]:
            nonlocal skip
            if skip:
                skip = False
                return []
            return self._steering_queue.drain()

        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            def _prepare_next_turn(ctx: PrepareNextTurnContext) -> Any:
                if self.prepare_next_turn_with_context is not None:
                    return self.prepare_next_turn_with_context(ctx)
                if self.prepare_next_turn is not None:
                    return cast(Any, self.prepare_next_turn)(self._abort)
                return None

            prepare_next_turn = _prepare_next_turn
        else:
            prepare_next_turn = None

        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=_maybe_async(self.transform_context),
            get_api_key=self.get_api_key,
            should_stop_after_turn=cast(Any, self.should_stop_after_turn),
            prepare_next_turn=prepare_next_turn,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            tool_execution=self.tool_execution,
            session_id=self.session_id,
            cache_retention=self.cache_retention,
            thinking_budgets=self.thinking_budgets,
            transport=self.transport,
            thinking_level=self._state.thinking_level,
            api_key=self.api_key,
            on_payload=self.on_payload,
            on_response=self.on_response,
            max_retry_delay_ms=self.max_retry_delay_ms,
            retry_policy=self.retry_policy,
            get_steering_messages=_get_steering,
            get_follow_up_messages=self._get_follow_up_messages,
        )

    async def _get_follow_up_messages(self) -> list[AgentMessage]:
        """AgentLoopConfig.get_follow_up_messages：消费 follow-up 队列。"""
        return self._follow_up_queue.drain()

    def _resolve_stream_fn(self) -> StreamFn:
        """解析 stream 函数：显式传入 > 全局默认。"""
        if self.stream_function is not None:
            return self.stream_function
        return get_default_stream_fn()

    # ------------------------------------------------------------------
    # 内部：事件处理
    # ------------------------------------------------------------------

    async def _process_event(self, event: AgentEvent) -> None:
        """事件 → 状态归约 + 扇出给订阅者。

        由 agent loop 在每个事件发生时调用。
        """
        event_type = event["type"]

        if event_type == "message_start":
            self._state.streaming_message = cast(AgentMessage, event.get("message"))

        elif event_type == "message_update":
            self._state.streaming_message = cast(AgentMessage, event.get("message"))

        elif event_type == "message_end":
            self._state.streaming_message = None
            self._state._append_message(cast(AgentMessage, event.get("message")))

        elif event_type == "tool_execution_start":
            self._state.pending_tool_calls.add(cast(str, event.get("tool_call_id")))

        elif event_type == "tool_execution_end":
            tc_id = cast(str, event.get("tool_call_id"))
            self._state.pending_tool_calls.discard(tc_id)

        elif event_type == "turn_end":
            msg = cast(AssistantMessage, event.get("message"))
            if msg.get("stop_reason") == "error":
                self._state.error_message = msg.get("error_message", "Unknown error")

        elif event_type == "agent_end":
            self._state.streaming_message = None

        # 扇出给订阅者：按订阅顺序 await，传递当前运行的 abort signal
        for listener in self._listeners:
            result = listener(event, self._abort)
            if result is not None:
                await result


# ============================================================================
# 辅助函数
# ============================================================================


def _normalize_input(
    input: str | AgentMessage | list[AgentMessage],
    images: list[ImageContent] | None = None,
) -> list[AgentMessage]:
    """标准化用户输入为 AgentMessage 列表。"""
    if isinstance(input, list):
        return list(input)

    if isinstance(input, str):
        content: list = [TextContent(type="text", text=input)]
        if images:
            content.extend(images)
        return [UserMessage(role="user", content=content, timestamp=now_ms())]

    # 单条 AgentMessage
    return [input]


def _default_convert_to_llm(
    messages: list[AgentMessage],
) -> list[Message]:
    """默认 AgentMessage → LLM Message 转换（agent 包 defaultConvertToLlm）。

    只透传 user / assistant / toolResult；其余 role（含 compactionSummary、
    bashExecution、custom 等）由应用层转换器处理
    （pi_agent._messages.convert_to_llm / pi_coding_agent.messages）。
    """
    result: list[Message] = []
    for m in messages:
        role = m.get("role", "")
        if role in ("user", "assistant", "toolResult"):
            result.append(cast(Message, m))
    return result


def _empty_usage() -> Usage:
    """空 usage（对齐 TS EMPTY_USAGE）。"""
    return {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def _maybe_async(fn: Callable | None) -> Any:
    """如果 fn 不是 async 函数，返回包装后的版本（可选）。"""
    # 最小核心：直接将同步函数传给 agent loop，
    # agent loop 内部通过 asyncio.iscoroutine 判断。
    return fn
