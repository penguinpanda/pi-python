"""
Agent 有状态包装类

拥有运行状态、管理并发、协调生命周期的有状态包装器。

核心职责:
- 互斥运行：同一 Agent 实例同时最多一个活跃运行
- 状态管理：维护 AgentState，通过 _process_event 归约事件 → 状态
- 生命周期管理：管理 abort signal + isStreaming 标志
- 事件订阅：subscribe() 注册监听器，按订阅顺序调用
- 桥接：AgentOptions → AgentLoopConfig → run_agent_loop()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pi_ai._types import (
    ImageContent,
    Message,
    Model,
    StreamOptions,
    TextContent,
    UserMessage,
    now_ms,
)
from pi_ai import RetryPolicy

from ._agent_loop import run_agent_loop, run_agent_loop_continue
from ._stream_fn import get_default_stream_fn
from ._types import (
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallResult,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
)

# ---------------------------------------------------------------------------
# AgentOptions
# ---------------------------------------------------------------------------


class AgentOptions:
    """Agent 构造选项（最小核心版）。

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
        transform_context: (
            Callable[[list[AgentMessage]], list[AgentMessage]] | None
        ) = None,
        get_api_key: Callable[[str], str | None] | None = None,
        before_tool_call: (
            Callable[
                [str, str, Any, AgentContext],
                BeforeToolCallResult | None,
            ]
            | None
        ) = None,
        after_tool_call: (
            Callable[
                [str, str, AgentToolResult, bool, AgentContext],
                AfterToolCallResult | None,
            ]
            | None
        ) = None,
        prepare_next_turn: (
            Callable[[AgentContext], Any] | None
        ) = None,
        should_stop_after_turn: Callable[[AgentContext], bool] | None = None,
        tool_execution: ToolExecutionMode = "sequential",
        # 重试策略。None = 默认启用（enabled=True, max_retries=3, base_delay_ms=2000）；
        # 传入 RetryPolicy(enabled=False) 可关闭重试。
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
        self.should_stop_after_turn = should_stop_after_turn
        self.tool_execution = tool_execution
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
            model=opts.model,  # type: ignore[arg-type]
            thinking_level=opts.thinking_level,
        )
        self._state.tools = opts.tools
        self._state.messages = opts.messages

        # -- 可配置钩子（公开属性，可在构造后修改） --
        self.convert_to_llm: Callable[[list[AgentMessage]], list[Message]] = (
            opts.convert_to_llm or _default_convert_to_llm
        )
        self.transform_context: (
            Callable[[list[AgentMessage]], list[AgentMessage]] | None
        ) = _maybe_async(opts.transform_context)
        self.stream_function: StreamFn | None = opts.stream_fn
        self.get_api_key: Callable[[str], str | None] | None = opts.get_api_key
        self.before_tool_call = opts.before_tool_call
        self.after_tool_call = opts.after_tool_call
        self.prepare_next_turn = opts.prepare_next_turn
        self.should_stop_after_turn = opts.should_stop_after_turn
        self.tool_execution: ToolExecutionMode = opts.tool_execution
        self.retry_policy: RetryPolicy | None = opts.retry_policy

        # -- 运行时 --
        self._active: bool = False
        self._abort: asyncio.Event | None = None
        self._listeners: list[Callable[[AgentEvent], None]] = []

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
            raise RuntimeError("Agent is already running. Use abort() to stop.")

        # 标准化输入
        prompts = _normalize_input(input, images)

        # 运行（prompts 由 run_agent_loop 注入到上下文并发出事件，
        # _process_event 处理 message_end 时会自动追加到 state.messages）
        await self._run_prompt(prompts)

    async def continue_(self) -> None:
        """从当前 transcript 继续（最后一条消息必须非 assistant）。"""
        if self._active:
            raise RuntimeError("Agent is already running. Use abort() to stop.")

        messages = self._state._messages
        if messages and messages[-1].get("role") == "assistant":
            raise RuntimeError(
                "Cannot continue: last message is an assistant message. "
                "Use prompt() instead."
            )

        await self._run_continue()

    def abort(self) -> None:
        """中止当前运行。"""
        if self._abort is not None:
            self._abort.set()

    def subscribe(
        self, listener: Callable[[AgentEvent], None]
    ) -> Callable[[], None]:
        """订阅生命周期事件。返回取消订阅函数。

        监听器按订阅顺序同步调用。
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    async def wait_for_idle(self) -> None:
        """等待当前运行结束（含所有事件监听器完成）。"""
        while self._active:
            await asyncio.sleep(0.01)

    def reset(self) -> None:
        """清空 transcript 和运行时状态。"""
        if self._active:
            raise RuntimeError("Cannot reset while agent is running.")
        self._state.messages = []
        self._state.streaming_message = None
        self._state.error_message = None
        self._state.pending_tool_calls = set()

    # ------------------------------------------------------------------
    # 内部：运行生命周期
    # ------------------------------------------------------------------

    async def _run_prompt(self, prompts: list[AgentMessage]) -> None:
        """执行一次完整的 prompt → agent loop 生命周期。"""
        self._active = True
        self._abort = asyncio.Event()
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state._messages),
            tools=list(self._state._tools),
        )

        try:
            await run_agent_loop(
                prompts=prompts,
                context=context,
                config=self._create_loop_config(),
                emit=self._process_event,
                signal=self._abort,
                stream_fn=self._resolve_stream_fn(),
            )
        except asyncio.CancelledError:
            # abort 触发，agent_end 事件已在 loop 中发出
            pass
        except Exception:
            # 意外异常 → 合成 agent_end
            await self._process_event({
                "type": "agent_end",
                "messages": list(self._state._messages),
            })
            raise
        finally:
            self._state.is_streaming = False
            self._active = False
            self._abort = None

    async def _run_continue(self) -> None:
        """从当前上下文继续。"""
        self._active = True
        self._abort = asyncio.Event()
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state._messages),
            tools=list(self._state._tools),
        )

        try:
            await run_agent_loop_continue(
                context=context,
                config=self._create_loop_config(),
                emit=self._process_event,
                signal=self._abort,
                stream_fn=self._resolve_stream_fn(),
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            await self._process_event({
                "type": "agent_end",
                "messages": list(self._state._messages),
            })
            raise
        finally:
            self._state.is_streaming = False
            self._active = False
            self._abort = None

    # ------------------------------------------------------------------
    # 内部：桥接
    # ------------------------------------------------------------------

    def _create_loop_config(self) -> AgentLoopConfig:
        """将 Agent 公开属性桥接为 AgentLoopConfig。"""
        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=_maybe_async(self.transform_context),
            get_api_key=self.get_api_key,
            should_stop_after_turn=self.should_stop_after_turn,
            prepare_next_turn=self.prepare_next_turn,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            tool_execution=self.tool_execution,
            retry_policy=self.retry_policy,
        )

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
            self._state.streaming_message = event["message"]

        elif event_type == "message_update":
            self._state.streaming_message = event["message"]

        elif event_type == "message_end":
            self._state.streaming_message = None
            self._state._append_message(event["message"])

        elif event_type == "tool_execution_start":
            self._state.pending_tool_calls.add(event["tool_call_id"])

        elif event_type == "tool_execution_end":
            tc_id = event["tool_call_id"]
            self._state.pending_tool_calls.discard(tc_id)

        elif event_type == "turn_end":
            msg = event["message"]
            if msg.get("stop_reason") == "error":
                self._state.error_message = msg.get("error_message", "Unknown error")

        elif event_type == "agent_end":
            self._state.streaming_message = None

        # 扇出给订阅者（同步调用）
        for listener in self._listeners:
            listener(event)


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
        if images:
            content: list = [TextContent(type="text", text=input)]
            content.extend(images)
            return [UserMessage(role="user", content=content, timestamp=now_ms())]
        return [UserMessage(role="user", content=input, timestamp=now_ms())]

    # 单条 AgentMessage
    return [input]


def _default_convert_to_llm(
    messages: list[AgentMessage],
) -> list[Message]:
    """默认 AgentMessage → LLM Message 转换：直接透传。

    过滤规则：移除没有 role 或不支持的 role 的消息。
    """
    result: list[Message] = []
    for m in messages:
        role = m.get("role", "")
        if role in ("system", "user", "assistant", "toolResult"):
            result.append(m)
    return result


def _maybe_async(fn: Callable | None) -> Any:
    """如果 fn 不是 async 函数，返回包装后的版本（可选）。"""
    # 最小核心：直接将同步函数传给 agent loop，
    # agent loop 内部通过 asyncio.iscoroutine 判断。
    return fn
