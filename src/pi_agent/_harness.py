"""AgentHarness 顶层协调器。

对齐 TS `harness/agent-harness.ts` 的核心结构：

- 拥有 DAG Session（默认内存存储；可注入 SessionStorage / Session）
- 阶段状态机（2.2）：idle → turn / compaction / branch_summary → idle，
  通过 `_active_tasks` 跟踪运行中的操作与配置变更
- 双事件系统（2.3）：`subscribe()` 通配符订阅 + `on()` 类型化 hook
  （顺序归约器：每个 handler 可替换前一个的输出）
- Save-point 安全模型（2.4）：运行期间的配置变更记录为 pending mutation，
  在 prepare_next_turn / turn_end / agent_end 处 flush，当前 LLM 请求不受影响
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    StreamOptions,
    TextContent,
    UserMessage,
    now_ms,
)

from ._agent_loop import run_agent_loop
from ._messages import convert_to_llm
from .branch_summarization import collect_entries_for_branch_summary, generate_branch_summary
from .compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    compact as run_compaction,
    prepare_compaction,
)
from .compaction_utils import apply_cache_first_truncation, estimate_cache_state
from .session.memory import InMemorySessionStorage
from .session.session import Session
from ._harness_types import (
    AbortEvent,
    AbortResult,
    AgentHarnessError,
    AgentHarnessEvent,
    AgentHarnessOptions,
    AgentHarnessPhase,
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    AgentHarnessSystemPrompt,
    BeforeAgentStartEvent,
    BeforeProviderRequestEvent,
    CompactResult,
    ContextEvent,
    ModelUpdateEvent,
    NavigateTreeResult,
    QueueUpdateEvent,
    ResourcesUpdateEvent,
    SavePointEvent,
    SessionBeforeCompactEvent,
    SessionBeforeTreeEvent,
    SessionCompactEvent,
    SessionTreeEvent,
    SettledEvent,
    Skill,
    ThinkingLevelUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsUpdateEvent,
    apply_stream_options_patch,
)
from ._stream_fn import get_default_stream_fn
from ._types import (
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    QueueMode,
    StreamFn,
    ThinkingLevel,
)


# ============================================================================
# 最小内存 Session（Phase 2 占位；Phase 3 替换为 DAG Session）
# ============================================================================


class AgentHarnessSession:
    """兼容占位（已废弃）：Phase 3 起 harness 使用 pi_agent.Session。

    保留导出以避免破坏已有引用；新代码请使用 pi_agent.Session。
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or f"session-{uuid4().hex[:12]}"
        self.messages: list[AgentMessage] = []

    def append_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def get_metadata(self) -> dict[str, Any]:
        return {"id": self.session_id}

    def get_branch(self) -> list[Any]:
        """Phase 3 DAG 前无分支概念。"""
        return []

    def get_leaf_id(self) -> str:
        """线性会话只有一个 leaf：会话根。"""
        return self.session_id

    def get_entry(self, entry_id: str) -> Any:
        """Phase 3 DAG 前无条目索引。"""
        return None


# ============================================================================
# 辅助
# ============================================================================


def _create_user_message(
    text: str,
    images: list[ImageContent] | None = None,
) -> UserMessage:
    content: list[Any] = [TextContent(type="text", text=text)]
    if images:
        content.extend(images)
    return UserMessage(role="user", content=content, timestamp=now_ms())


def _create_failure_message(
    model: Any,
    error: BaseException | None,
    aborted: bool,
) -> AssistantMessage:
    message = str(error) if error is not None else "Unknown error"
    if error is not None and isinstance(error, Exception) and not error.args:
        message = type(error).__name__
    return {
        "role": "assistant",
        "content": [TextContent(type="text", text="")],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "stop_reason": "aborted" if aborted else "error",
        "error_message": message,
        "timestamp": now_ms(),
        "usage": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 0,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
    }


def _find_duplicates(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _normalize_harness_error(
    error: BaseException | None,
    fallback_code: str,
) -> AgentHarnessError:
    if isinstance(error, AgentHarnessError):
        return error
    return AgentHarnessError(
        cast(Any, fallback_code),
        str(error) if error is not None else "Unknown error",
        cause=error,
    )


def _format_skill_invocation(
    skill: Skill,
    additional_instructions: str | None = None,
) -> str:
    """格式化技能调用（简化版，Phase 4.4 将移植完整 skills 系统）。"""
    directory = os.path.dirname(skill.file_path).replace("\\", "/")
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {directory}.\n\n{skill.content}\n</skill>"
    )
    if additional_instructions:
        return f"{skill_block}\n\n{additional_instructions}"
    return skill_block


def _format_template_invocation(content: str, args: list[str]) -> str:
    """模板参数替换（简化版，Phase 4.5 将移植完整模板系统）。

    支持 $1..$9、$@ / $ARGUMENTS、${@:N}、${@:N:L}（1-based）。
    """
    result = content
    for index, arg in enumerate(args, start=1):
        result = result.replace(f"${index}", arg)
    joined = "\n".join(args)
    result = result.replace("$@", joined).replace("$ARGUMENTS", joined)

    def _slice(match: re.Match[str]) -> str:
        start = int(match.group(1)) - 1
        end = int(match.group(2)) if match.group(2) else len(args)
        return "\n".join(args[start:end])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", _slice, result)
    return result


class _TurnState:
    """一次 turn 的不可变快照（对齐 TS AgentHarnessTurnState）。"""

    __slots__ = (
        "messages",
        "resources",
        "tool_context",
        "stream_options",
        "session_id",
        "system_prompt",
        "model",
        "thinking_level",
        "tools",
        "active_tools",
    )

    def __init__(
        self,
        messages: list[AgentMessage],
        resources: AgentHarnessResources,
        tool_context: Any,
        stream_options: AgentHarnessStreamOptions,
        session_id: str,
        system_prompt: str,
        model: Any,
        thinking_level: ThinkingLevel,
        tools: list[AgentTool],
        active_tools: list[AgentTool],
    ) -> None:
        self.messages = messages
        self.resources = resources
        self.tool_context = tool_context
        self.stream_options = stream_options
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.model = model
        self.thinking_level = thinking_level
        self.tools = tools
        self.active_tools = active_tools


_SUBSCRIBER_EVENT_TYPE = "*"


# ============================================================================
# AgentHarness
# ============================================================================


class AgentHarness:
    """Agent 应用的顶层协调器（Phase 2 骨架）。"""

    def __init__(self, options: AgentHarnessOptions) -> None:
        self._session: Any = self._resolve_session(options.session)
        self._compaction_settings: CompactionSettings = (
            options.compaction_settings
            if options.compaction_settings is not None
            else DEFAULT_COMPACTION_SETTINGS
        )
        self._model = options.model
        self._thinking_level: ThinkingLevel = options.thinking_level
        self._system_prompt_src: AgentHarnessSystemPrompt | None = options.system_prompt
        self._resources: AgentHarnessResources = (
            options.resources.clone() if options.resources is not None else AgentHarnessResources()
        )
        self._tool_context = options.tool_context
        self._stream_fn: StreamFn | None = options.stream_fn
        self._stream_options: AgentHarnessStreamOptions = (
            options.stream_options.clone()
            if options.stream_options is not None
            else AgentHarnessStreamOptions()
        )

        # 工具注册表 + 激活列表
        self._tools: dict[str, AgentTool] = {}
        for tool in options.tools or []:
            self._tools[tool.name] = tool
        self._active_tool_names: list[str] = (
            list(options.active_tool_names)
            if options.active_tool_names is not None
            else [tool.name for tool in options.tools or []]
        )
        self._validate_tool_names(self._active_tool_names)

        # 消息队列（steer / follow-up / next-turn）
        self._steer_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._next_turn_queue: list[AgentMessage] = []
        self._steering_mode: QueueMode = options.steering_mode
        self._follow_up_mode: QueueMode = options.follow_up_mode

        # 阶段状态机 + 任务跟踪（2.2）
        self._phase: AgentHarnessPhase = "idle"
        self._active_tasks: dict[asyncio.Future[None], str] = {}
        self._abort_event: asyncio.Event | None = None

        # Save-point（2.4）：运行期间的变更先记 pending，边界处 flush
        self._pending_mutations: int = 0
        self._pending_message_writes: list[AgentMessage] = []

        # 生命周期
        self._is_shutdown = False
        self._shutdown_task: asyncio.Task | None = None

        # 双事件系统（2.3）
        self._handlers: dict[str, set[Callable[..., Any]]] = {}

    # ------------------------------------------------------------------
    # 生命周期守卫
    # ------------------------------------------------------------------

    def _assert_not_shut_down(self) -> None:
        if self._is_shutdown:
            raise AgentHarnessError("invalid_state", "AgentHarness has been shut down")

    def _resolve_session(self, session: Any) -> Any:
        if session is None:
            return Session(InMemorySessionStorage())
        if isinstance(session, AgentHarnessSession):
            raise AgentHarnessError(
                "invalid_argument",
                "AgentHarnessSession is deprecated; pass pi_agent.Session or a SessionStorage",
            )
        if isinstance(session, Session):
            return session
        if hasattr(session, "get_metadata") and hasattr(session, "append_entry"):
            return Session(session)
        raise AgentHarnessError(
            "invalid_argument",
            "session must be pi_agent.Session or a SessionStorage",
        )

    async def _get_session_id(self) -> str:
        metadata = await self._session.get_metadata()
        return str(metadata["id"])

    # ------------------------------------------------------------------
    # 任务跟踪 / 阶段状态机（2.2）
    # ------------------------------------------------------------------

    async def _track(self, kind: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        """跟踪一个操作/变更，直到其 settle。"""
        settle: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_tasks[settle] = kind
        try:
            return await operation()
        finally:
            self._active_tasks.pop(settle, None)
            if not settle.done():
                settle.set_result(None)

    async def _wait_for_tasks(self, kind: str | None = None) -> None:
        """等待指定类型的任务全部 settle（循环处理运行期间新注册的任务）。"""
        while True:
            tasks = [
                future
                for future, task_kind in self._active_tasks.items()
                if kind is None or task_kind == kind
            ]
            if not tasks:
                return
            await asyncio.wait(tasks)

    def _start_operation(self) -> tuple[asyncio.Event, Callable[[], None]]:
        """开始一个操作（prompt/compact/navigateTree）：返回 abort 信号与 finish 回调。"""
        abort_event = asyncio.Event()
        finish_event: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_tasks[finish_event] = "operation"
        self._abort_event = abort_event

        def _finish() -> None:
            self._abort_event = None
            self._active_tasks.pop(finish_event, None)
            if not finish_event.done():
                finish_event.set_result(None)

        return abort_event, _finish

    async def wait_for_idle(self) -> None:
        """等待当前操作（turn/compaction/branch_summary）结束。"""
        await self._wait_for_tasks("operation")

    # ------------------------------------------------------------------
    # 双事件系统（2.3）
    # ------------------------------------------------------------------

    def subscribe(
        self,
        listener: Callable[[AgentHarnessEvent, asyncio.Event | None], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """通配符订阅：接收全部 harness 事件（agent loop 事件 + harness 自有事件）。"""
        self._assert_not_shut_down()
        return self._add_handler(_SUBSCRIBER_EVENT_TYPE, listener)

    def on(
        self,
        event_type: str,
        handler: Callable[[Any], Any],
    ) -> Callable[[], None]:
        """类型化 hook：顺序归约器，后一个 handler 可替换前一个的输出。"""
        self._assert_not_shut_down()
        return self._add_handler(event_type, handler)

    def _add_handler(self, event_type: str, handler: Callable[..., Any]) -> Callable[[], None]:
        handlers = self._handlers.setdefault(event_type, set())
        handlers.add(handler)

        def _unsubscribe() -> None:
            handlers.discard(handler)

        return _unsubscribe

    async def _emit_any(
        self, event: AgentHarnessEvent, signal: asyncio.Event | None = None
    ) -> None:
        for listener in list(self._handlers.get(_SUBSCRIBER_EVENT_TYPE, ())):
            try:
                result = listener(event, signal)
                if result is not None:
                    await result
            except BaseException as error:
                raise _normalize_harness_error(error, "hook") from error

    async def _emit_own(self, event: Any) -> None:
        await self._emit_any(cast(AgentHarnessEvent, event))

    async def _emit_hook(self, event_type: str, event: Any) -> Any:
        """类型化 hook：按注册顺序调用，返回最后一个非 None 结果。"""
        handlers = self._handlers.get(event_type)
        if not handlers:
            return None
        last_result: Any = None
        for handler in list(handlers):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    last_result = result
            except BaseException as error:
                raise _normalize_harness_error(error, "hook") from error
        return last_result

    # ------------------------------------------------------------------
    # Turn 状态
    # ------------------------------------------------------------------

    def _get_resources(self) -> AgentHarnessResources:
        return self._resources.clone()

    async def _resolve_tool_context(self) -> Any:
        if callable(self._tool_context):
            return await self._tool_context()
        return self._tool_context

    def _bind_tool(self, tool: AgentTool, context: Any) -> AgentTool:
        """把应用工具（execute 接收 context 作第 5 参）绑定到当前 turn 的 context。"""
        original_execute = tool.execute

        async def _execute(
            tool_call_id: str,
            params: Any,
            signal: asyncio.Event | None = None,
            on_update: Any = None,
        ) -> AgentToolResult:
            return await original_execute(tool_call_id, params, signal, on_update, context)

        bound = AgentTool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            label=tool.label,
            execute=_execute,
            execution_mode=tool.execution_mode,
        )
        bound.before_execute = tool.before_execute
        bound.after_execute = tool.after_execute
        return bound

    def _create_context(
        self,
        turn_state: _TurnState,
        system_prompt: str | None = None,
    ) -> AgentContext:
        """构造 loop 上下文：激活工具绑定当前 turn 的 tool_context。"""
        return AgentContext(
            system_prompt=system_prompt or turn_state.system_prompt,
            messages=list(turn_state.messages),
            tools=[
                self._bind_tool(tool, turn_state.tool_context) for tool in turn_state.active_tools
            ],
        )

    async def _create_turn_state(self) -> _TurnState:
        self._assert_not_shut_down()
        context = await self._session.build_context()
        messages = list(context["messages"])
        if self._compaction_settings.cache_first and estimate_cache_state(messages) != "cold":
            messages = apply_cache_first_truncation(
                messages,
                context_window=self._model.context_window if self._model is not None else 0,
                reserve_tokens=self._compaction_settings.reserve_tokens,
                protect_recent_tokens=self._compaction_settings.protect_recent_tokens,
            )
        resources = self._get_resources()
        tool_context = await self._resolve_tool_context()
        tools = list(self._tools.values())
        active_tools = [
            self._tools[name] for name in self._active_tool_names if name in self._tools
        ]
        system_prompt = "You are a helpful assistant."
        if isinstance(self._system_prompt_src, str):
            system_prompt = self._system_prompt_src
        elif callable(self._system_prompt_src):
            prompt_result = self._system_prompt_src(
                {
                    "session": self._session,
                    "model": self._model,
                    "thinking_level": self._thinking_level,
                    "active_tools": list(active_tools),
                    "resources": resources,
                }
            )
            if asyncio.iscoroutine(prompt_result):
                system_prompt = await cast(Awaitable[str], prompt_result)
            else:
                system_prompt = cast(str, prompt_result)
        return _TurnState(
            messages=messages,
            resources=resources,
            tool_context=tool_context,
            stream_options=self._stream_options.clone(),
            session_id=await self._get_session_id(),
            system_prompt=system_prompt,
            model=self._model,
            thinking_level=self._thinking_level,
            tools=tools,
            active_tools=active_tools,
        )

    # ------------------------------------------------------------------
    # Loop 接线
    # ------------------------------------------------------------------

    async def _apply_request_options(
        self,
        model: Any,
        session_id: str,
        options: Any,
    ) -> dict[str, Any]:
        request_options = await self._emit_before_provider_request(
            model,
            session_id,
            self._stream_options.clone(),
        )
        merged = dict(options or {})
        if request_options.max_retries is not None:
            merged["max_retries"] = request_options.max_retries
        if request_options.max_retry_delay_ms is not None:
            merged["max_retry_delay_ms"] = request_options.max_retry_delay_ms
        if request_options.headers is not None:
            merged["headers"] = dict(request_options.headers)
        if request_options.cache_retention is not None:
            merged["cache_retention"] = request_options.cache_retention
        if request_options.transport is not None:
            merged["transport"] = request_options.transport
        if self._thinking_level != "off":
            merged["reasoning"] = self._thinking_level
        return merged

    def _create_stream_fn(self, get_turn_state: Callable[[], _TurnState]) -> StreamFn:
        base = self._stream_fn or get_default_stream_fn()

        async def _stream(model: Any, context: Any, options: Any = None) -> Any:
            turn_state = get_turn_state()
            merged = await self._apply_request_options(model, turn_state.session_id, options)
            return await base(model, context, cast(StreamOptions, merged))

        return _stream

    def _create_summary_stream_fn(self) -> StreamFn:
        base = self._stream_fn or get_default_stream_fn()

        async def _stream(model: Any, context: Any, options: Any = None) -> Any:
            merged = await self._apply_request_options(model, await self._get_session_id(), options)
            return await base(model, context, cast(StreamOptions, merged))

        return _stream

    async def _emit_before_provider_request(
        self,
        model: Any,
        session_id: str,
        stream_options: AgentHarnessStreamOptions,
    ) -> AgentHarnessStreamOptions:
        current = stream_options.clone()
        handlers = self._handlers.get("before_provider_request")
        if not handlers:
            return current
        for handler in list(handlers):
            try:
                result = handler(
                    BeforeProviderRequestEvent(
                        type="before_provider_request",
                        model=model,
                        session_id=session_id,
                        stream_options=current.clone(),
                    )
                )
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None and result.stream_options is not None:
                    current = apply_stream_options_patch(current, result.stream_options)
            except BaseException as error:
                raise _normalize_harness_error(error, "hook") from error
        return current

    async def _transform_context(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        result = await self._emit_hook(
            "context", ContextEvent(type="context", messages=list(messages))
        )
        if result is not None and result.messages is not None:
            return list(result.messages)
        return messages

    async def _before_tool_call(self, ctx: Any) -> BeforeToolCallResult | None:
        result = await self._emit_hook(
            "tool_call",
            ToolCallEvent(
                type="tool_call",
                tool_call_id=ctx.tool_call["id"],
                tool_name=ctx.tool_call["name"],
                input=dict(ctx.args) if isinstance(ctx.args, dict) else {},
            ),
        )
        if result is None:
            return None
        return BeforeToolCallResult(block=result.block, reason=result.reason)

    async def _after_tool_call(self, ctx: Any) -> AfterToolCallResult | None:
        result = await self._emit_hook(
            "tool_result",
            ToolResultEvent(
                type="tool_result",
                tool_call_id=ctx.tool_call["id"],
                tool_name=ctx.tool_call["name"],
                input=dict(ctx.args) if isinstance(ctx.args, dict) else {},
                content=list(ctx.result.content),
                details=ctx.result.details,
                is_error=ctx.is_error,
                usage=ctx.result.usage,
            ),
        )
        if result is None:
            return None
        return AfterToolCallResult(
            content=result.content,
            details=result.details,
            is_error=result.is_error,
            usage=result.usage,
            terminate=result.terminate,
        )

    async def _prepare_next_turn(
        self,
        get_turn_state: Callable[[], _TurnState],
        set_turn_state: Callable[[_TurnState], None],
        turn_context: AgentContext,
    ) -> AgentLoopTurnUpdate:
        """Save-point（2.4）：turn 边界重建状态，配置变更在此之后生效。"""
        await self._flush_pending_writes()
        next_state = await self._create_turn_state()
        set_turn_state(next_state)
        return AgentLoopTurnUpdate(
            context=self._create_context(next_state),
            model=next_state.model,
            thinking_level=next_state.thinking_level,
        )

    def _create_loop_config(
        self,
        get_turn_state: Callable[[], _TurnState],
        set_turn_state: Callable[[_TurnState], None],
    ) -> AgentLoopConfig:
        turn_state = get_turn_state()
        return AgentLoopConfig(
            model=turn_state.model,
            convert_to_llm=convert_to_llm,
            transform_context=self._transform_context,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            prepare_next_turn=lambda ctx: self._prepare_next_turn(
                get_turn_state, set_turn_state, ctx.context
            ),
            get_steering_messages=self._drain_steering,
            get_follow_up_messages=self._drain_follow_up,
            session_id=turn_state.session_id,
            cache_retention=turn_state.stream_options.cache_retention,
        )

    # ------------------------------------------------------------------
    # 队列
    # ------------------------------------------------------------------

    async def _emit_queue_update(self) -> None:
        await self._emit_own(
            QueueUpdateEvent(
                type="queue_update",
                steer=list(self._steer_queue),
                follow_up=list(self._follow_up_queue),
                next_turn=list(self._next_turn_queue),
            )
        )

    async def _drain_queue(
        self,
        queue: list[AgentMessage],
        mode: QueueMode,
    ) -> list[AgentMessage]:
        if mode == "all":
            messages = list(queue)
            queue.clear()
        else:
            messages = queue[:1]
            del queue[:1]
        if not messages:
            return messages
        try:
            await self._emit_queue_update()
            return messages
        except BaseException as error:
            queue[:0] = messages
            raise _normalize_harness_error(error, "hook") from error

    async def _drain_steering(self) -> list[AgentMessage]:
        return await self._drain_queue(self._steer_queue, self._steering_mode)

    async def _drain_follow_up(self) -> list[AgentMessage]:
        return await self._drain_queue(self._follow_up_queue, self._follow_up_mode)

    # ------------------------------------------------------------------
    # Save-point（2.4）
    # ------------------------------------------------------------------

    async def _flush_pending_writes(self) -> None:
        while self._pending_message_writes:
            message = self._pending_message_writes.pop(0)
            await self._session.append_message(message)
        self._pending_mutations = 0

    # ------------------------------------------------------------------
    # Agent 事件转发
    # ------------------------------------------------------------------

    async def _handle_agent_event(
        self,
        event: AgentEvent,
        signal: asyncio.Event | None = None,
    ) -> None:
        if signal is None:
            signal = self._abort_event
        event_type = event["type"]

        if event_type == "message_end":
            await self._session.append_message(cast(AgentMessage, event.get("message")))
            await self._emit_any(event, signal)
            return

        if event_type == "turn_end":
            event_error: BaseException | None = None
            try:
                await self._emit_any(event, signal)
            except BaseException as error:
                event_error = error
            had_pending = self._pending_mutations > 0 or bool(self._pending_message_writes)
            await self._flush_pending_writes()
            await self._emit_own(
                SavePointEvent(
                    type="save_point",
                    had_pending_mutations=had_pending,
                )
            )
            if event_error is not None:
                raise event_error
            return

        if event_type == "agent_end":
            await self._flush_pending_writes()
            self._phase = "idle"
            await self._emit_any(event, signal)
            await self._emit_own(
                SettledEvent(
                    type="settled",
                    next_turn_count=len(self._next_turn_queue),
                )
            )
            return

        await self._emit_any(event, signal)

    async def _emit_run_failure(
        self,
        model: Any,
        error: BaseException | None,
        signal: asyncio.Event | None,
    ) -> list[AgentMessage]:
        aborted = signal is not None and signal.is_set()
        failure_message = _create_failure_message(model, error, aborted)
        await self._handle_agent_event(
            {"type": "message_start", "message": failure_message}, signal
        )
        await self._handle_agent_event({"type": "message_end", "message": failure_message}, signal)
        await self._handle_agent_event(
            {"type": "turn_end", "message": failure_message, "tool_results": []}, signal
        )
        await self._handle_agent_event({"type": "agent_end", "messages": [failure_message]}, signal)
        return [failure_message]

    # ------------------------------------------------------------------
    # 工具校验
    # ------------------------------------------------------------------

    def _validate_tool_names(
        self,
        tool_names: list[str],
        tools: dict[str, AgentTool] | None = None,
    ) -> None:
        registry = tools if tools is not None else self._tools
        duplicates = _find_duplicates(tool_names)
        if duplicates:
            raise AgentHarnessError(
                "invalid_argument",
                f"Duplicate active tool name(s): {', '.join(duplicates)}",
            )
        missing = [name for name in tool_names if name not in registry]
        if missing:
            raise AgentHarnessError(
                "invalid_argument",
                f"Unknown tool(s): {', '.join(missing)}",
            )

    # ------------------------------------------------------------------
    # 操作接口
    # ------------------------------------------------------------------

    async def _execute_turn(
        self,
        turn_state: _TurnState,
        text: str,
        signal: asyncio.Event,
        images: list[ImageContent] | None = None,
    ) -> AssistantMessage:
        self._assert_not_shut_down()
        active_turn_state = turn_state
        messages: list[AgentMessage] = [_create_user_message(text, images)]

        # nextTurn 队列消息先于本次 prompt 注入
        if self._next_turn_queue:
            queued = list(self._next_turn_queue)
            self._next_turn_queue = []
            try:
                await self._emit_queue_update()
            except BaseException as error:
                self._next_turn_queue = queued + self._next_turn_queue
                raise _normalize_harness_error(error, "hook") from error
            messages = queued + messages

        before_result = await self._emit_hook(
            "before_agent_start",
            BeforeAgentStartEvent(
                type="before_agent_start",
                prompt=text,
                images=images,
                system_prompt=turn_state.system_prompt,
                resources=turn_state.resources,
            ),
        )
        self._assert_not_shut_down()
        if before_result is not None and before_result.messages:
            messages = messages + list(before_result.messages)
        system_prompt = turn_state.system_prompt
        if before_result is not None and before_result.system_prompt:
            system_prompt = before_result.system_prompt

        def _get_turn_state() -> _TurnState:
            return active_turn_state

        def _set_turn_state(next_state: _TurnState) -> None:
            nonlocal active_turn_state
            active_turn_state = next_state

        try:
            new_messages = await run_agent_loop(
                prompts=messages,
                context=self._create_context(turn_state, system_prompt),
                config=self._create_loop_config(_get_turn_state, _set_turn_state),
                emit=self._handle_agent_event,
                signal=signal,
                stream_fn=self._create_stream_fn(_get_turn_state),
            )
        except BaseException as error:
            return cast(
                AssistantMessage,
                (await self._emit_run_failure(turn_state.model, error, signal))[-1],
            )
        finally:
            await self._flush_pending_writes()

        for message in reversed(new_messages):
            if message.get("role") == "assistant":
                return cast(AssistantMessage, message)
        raise AgentHarnessError(
            "invalid_state",
            "AgentHarness prompt completed without an assistant message",
        )

    async def prompt(
        self,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> AssistantMessage:
        """发送用户消息，运行完整 agent loop。返回最后一条 assistant 消息。"""
        self._assert_not_shut_down()
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        abort_event, finish = self._start_operation()
        try:
            turn_state = await self._create_turn_state()
            return await self._execute_turn(turn_state, text, abort_event, images)
        except BaseException as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish()

    async def skill(
        self,
        name: str,
        additional_instructions: str | None = None,
    ) -> AssistantMessage:
        """以技能调用方式运行一轮。"""
        self._assert_not_shut_down()
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        abort_event, finish = self._start_operation()
        try:
            turn_state = await self._create_turn_state()
            skill = next(
                (s for s in (turn_state.resources.skills or []) if s.name == name),
                None,
            )
            if skill is None:
                raise AgentHarnessError("invalid_argument", f"Unknown skill: {name}")
            return await self._execute_turn(
                turn_state,
                _format_skill_invocation(skill, additional_instructions),
                abort_event,
            )
        except BaseException as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish()

    async def prompt_from_template(
        self,
        name: str,
        args: list[str] | None = None,
    ) -> AssistantMessage:
        """以提示模板调用方式运行一轮。"""
        self._assert_not_shut_down()
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        abort_event, finish = self._start_operation()
        try:
            turn_state = await self._create_turn_state()
            template = next(
                (t for t in (turn_state.resources.prompt_templates or []) if t.name == name),
                None,
            )
            if template is None:
                raise AgentHarnessError("invalid_argument", f"Unknown prompt template: {name}")
            return await self._execute_turn(
                turn_state,
                _format_template_invocation(template.content, args or []),
                abort_event,
            )
        except BaseException as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish()

    async def steer(
        self,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> None:
        """运行中注入引导消息（趁 agent 还在工作时调整方向）。"""
        self._assert_not_shut_down()
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot steer while idle")
        self._steer_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def follow_up(
        self,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Agent 即将停止时注入后续消息。"""
        self._assert_not_shut_down()
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot follow up while idle")
        self._follow_up_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def next_turn(
        self,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> None:
        """排队一条消息，在下次 prompt() 时先于新 prompt 注入。"""
        self._assert_not_shut_down()
        self._next_turn_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def append_message(self, message: AgentMessage) -> None:
        """向会话追加一条消息（运行中记 pending，agent_end 时 flush）。"""
        self._assert_not_shut_down()
        await self._track("mutation", lambda: self._apply_append_message(message))

    async def _apply_append_message(self, message: AgentMessage) -> None:
        if self._phase == "idle":
            await self._session.append_message(message)
        else:
            self._pending_message_writes.append(message)

    async def compact(self, custom_instructions: str | None = None) -> CompactResult:
        """上下文压缩：prepare → hook → LLM 摘要 → 写入 compaction 条目。"""
        self._assert_not_shut_down()
        if self._phase != "idle":
            raise AgentHarnessError("busy", "compact() requires idle harness")
        self._phase = "compaction"
        abort_event, finish = self._start_operation()
        try:
            branch_entries = await self._session.get_branch()
            ok_flag, preparation = prepare_compaction(branch_entries, self._compaction_settings)
            if not ok_flag:
                raise _normalize_harness_error(cast(BaseException, preparation), "compaction")
            hook_result = await self._emit_hook(
                "session_before_compact",
                SessionBeforeCompactEvent(
                    type="session_before_compact",
                    preparation=preparation,
                    branch_entries=branch_entries,
                    custom_instructions=custom_instructions,
                ),
            )
            if hook_result is not None and hook_result.cancel:
                raise AgentHarnessError("compaction", "Compaction cancelled")
            provided = hook_result.compaction if hook_result is not None else None
            if provided is not None:
                await self._session.append_compaction(
                    provided.summary,
                    provided.first_kept_entry_id,
                    provided.tokens_before,
                    details=provided.details,
                    from_hook=True,
                    usage=provided.usage,
                    retained_tail=provided.retained_tail,
                )
                await self._emit_own(SessionCompactEvent(type="session_compact", from_hook=True))
                return provided
            if preparation is None:
                raise AgentHarnessError("compaction", "Nothing to compact")
            ok_flag, result = await run_compaction(
                preparation,
                self._create_summary_stream_fn(),
                self._model,
                custom_instructions=custom_instructions,
                signal=abort_event,
                thinking_level=self._thinking_level,
            )
            if not ok_flag:
                raise _normalize_harness_error(cast(BaseException, result), "compaction")
            await self._session.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                details=result.details,
                from_hook=False,
                usage=result.usage,
                retained_tail=result.retained_tail,
            )
            await self._emit_own(SessionCompactEvent(type="session_compact", from_hook=False))
            return CompactResult(
                summary=result.summary,
                first_kept_entry_id=result.first_kept_entry_id,
                tokens_before=result.tokens_before,
                usage=result.usage,
                retained_tail=result.retained_tail,
                details=result.details,
            )
        except BaseException as error:
            raise _normalize_harness_error(error, "compaction") from error
        finally:
            self._phase = "idle"
            finish()

    async def navigate_tree(self, target_id: str, *, summarize: bool = False) -> NavigateTreeResult:
        """分支导航：hook → （可选 LLM 分支摘要）→ move leaf → 写 branch_summary 条目。"""
        self._assert_not_shut_down()
        if self._phase != "idle":
            raise AgentHarnessError("busy", "navigateTree() requires idle harness")
        self._phase = "branch_summary"
        abort_event, finish = self._start_operation()
        try:
            old_leaf_id = await self._session.get_leaf_id()
            if old_leaf_id == target_id:
                return NavigateTreeResult(cancelled=False)
            if await self._session.get_entry(target_id) is None:
                raise AgentHarnessError("invalid_argument", f"Entry {target_id} not found")
            hook_result = await self._emit_hook(
                "session_before_tree",
                SessionBeforeTreeEvent(
                    type="session_before_tree",
                    target_id=target_id,
                    old_leaf_id=old_leaf_id,
                    summarize=summarize,
                    custom_instructions=None,
                    label=None,
                ),
            )
            if hook_result is not None and hook_result.cancel:
                return NavigateTreeResult(cancelled=True)

            summary_data: dict[str, Any] | None = None
            from_hook = False
            if hook_result is not None and hook_result.summary is not None:
                summary_data = {
                    "summary": str(hook_result.summary),
                    "fromHook": True,
                }
                from_hook = True
            elif summarize:
                custom_instructions = (
                    hook_result.custom_instructions if hook_result is not None else None
                )
                replace_instructions = bool(
                    hook_result.replace_instructions if hook_result is not None else False
                )
                collected = await collect_entries_for_branch_summary(
                    self._session, old_leaf_id, target_id
                )
                ok_flag, summary = await generate_branch_summary(
                    collected["entries"],
                    stream_fn=self._create_summary_stream_fn(),
                    model=self._model,
                    signal=abort_event,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    reserve_tokens=self._compaction_settings.reserve_tokens,
                )
                if not ok_flag:
                    raise _normalize_harness_error(cast(BaseException, summary), "branch_summary")
                summary_data = {
                    "summary": summary["summary"],
                    "details": {
                        "readFiles": summary.get("readFiles", []),
                        "modifiedFiles": summary.get("modifiedFiles", []),
                    },
                    "usage": summary.get("usage"),
                    "fromHook": False,
                }

            summary_entry_id = await self._session.move_to(target_id, summary_data)
            label = hook_result.label if hook_result is not None else None
            if label is not None:
                await self._session.append_label(target_id, label)
            summary_entry = (
                await self._session.get_entry(summary_entry_id)
                if summary_entry_id is not None
                else None
            )
            new_leaf_id = await self._session.get_leaf_id()
            await self._emit_own(
                SessionTreeEvent(
                    type="session_tree",
                    new_leaf_id=new_leaf_id or target_id,
                    old_leaf_id=old_leaf_id,
                    from_hook=from_hook,
                )
            )
            return NavigateTreeResult(cancelled=False, summary_entry=summary_entry)
        except BaseException as error:
            raise _normalize_harness_error(error, "branch_summary") from error
        finally:
            self._phase = "idle"
            finish()

    async def abort(self) -> AbortResult:
        """中止当前操作并清空 steer/follow-up 队列。"""
        self._assert_not_shut_down()
        cleared_steer = list(self._steer_queue)
        cleared_follow_up = list(self._follow_up_queue)
        self._steer_queue = []
        self._follow_up_queue = []
        if self._abort_event is not None:
            self._abort_event.set()
        errors: list[BaseException] = []
        try:
            await self._emit_queue_update()
        except BaseException as error:
            errors.append(error)
        try:
            await self.wait_for_idle()
        except BaseException as error:
            errors.append(error)
        try:
            await self._emit_own(
                AbortEvent(
                    type="abort",
                    cleared_steer=cleared_steer,
                    cleared_follow_up=cleared_follow_up,
                )
            )
        except BaseException as error:
            errors.append(error)
        if errors:
            raise AgentHarnessError("hook", "Abort completed with errors", cause=errors[0])
        return AbortResult(cleared_steer=cleared_steer, cleared_follow_up=cleared_follow_up)

    async def shutdown(self) -> None:
        """永久停止本 harness：清空队列、中止当前操作并等待全部任务 settle。"""
        if self._shutdown_task is not None:
            return await self._shutdown_task
        self._is_shutdown = True
        self._steer_queue = []
        self._follow_up_queue = []
        self._next_turn_queue = []
        self._pending_message_writes = []
        if self._abort_event is not None:
            self._abort_event.set()
        self._shutdown_task = asyncio.create_task(self._wait_for_tasks())
        return await self._shutdown_task

    # ------------------------------------------------------------------
    # 配置方法（Save-point：运行中的变更记 pending，边界生效）
    # ------------------------------------------------------------------

    def get_model(self) -> Any:
        return self._model

    async def get_leaf_id(self) -> str | None:
        """当前 leaf 条目 ID（DAG 会话）。"""
        return await self._session.get_leaf_id()

    def get_compaction_settings(self) -> CompactionSettings:
        return self._compaction_settings

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        self._assert_not_shut_down()
        self._compaction_settings = settings

    async def set_model(self, model: Any) -> None:
        self._assert_not_shut_down()
        await self._track("mutation", lambda: self._apply_model(model))

    async def _apply_model(self, model: Any) -> None:
        previous_model = self._model
        if self._phase != "idle":
            self._pending_mutations += 1
        self._model = model
        await self._emit_own(
            ModelUpdateEvent(
                type="model_update",
                model=model,
                previous_model=previous_model,
                source="set",
            )
        )

    def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._assert_not_shut_down()
        await self._track("mutation", lambda: self._apply_thinking_level(level))

    async def _apply_thinking_level(self, level: ThinkingLevel) -> None:
        previous_level = self._thinking_level
        if self._phase != "idle":
            self._pending_mutations += 1
        self._thinking_level = level
        await self._emit_own(
            ThinkingLevelUpdateEvent(
                type="thinking_level_update",
                level=level,
                previous_level=previous_level,
            )
        )

    def get_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    async def set_tools(
        self,
        tools: list[AgentTool],
        active_tool_names: list[str] | None = None,
    ) -> None:
        self._assert_not_shut_down()
        await self._track(
            "mutation",
            lambda: self._apply_tools(tools, active_tool_names),
        )

    async def _apply_tools(
        self,
        tools: list[AgentTool],
        active_tool_names: list[str] | None,
    ) -> None:
        duplicates = _find_duplicates([tool.name for tool in tools])
        if duplicates:
            raise AgentHarnessError(
                "invalid_argument",
                f"Duplicate tool name(s): {', '.join(duplicates)}",
            )
        next_tools = {tool.name: tool for tool in tools}
        next_active = (
            list(active_tool_names)
            if active_tool_names is not None
            else list(self._active_tool_names)
        )
        self._validate_tool_names(next_active, next_tools)
        previous_names = list(self._tools.keys())
        previous_active = list(self._active_tool_names)
        if self._phase != "idle":
            self._pending_mutations += 1
        self._tools = next_tools
        self._active_tool_names = next_active
        await self._emit_own(
            ToolsUpdateEvent(
                type="tools_update",
                tool_names=list(self._tools.keys()),
                previous_tool_names=previous_names,
                active_tool_names=list(self._active_tool_names),
                previous_active_tool_names=previous_active,
                source="set",
            )
        )

    def get_active_tools(self) -> list[AgentTool]:
        return [self._tools[name] for name in self._active_tool_names if name in self._tools]

    async def set_active_tools(self, tool_names: list[str]) -> None:
        self._assert_not_shut_down()
        await self._track("mutation", lambda: self._apply_active_tools(tool_names))

    async def _apply_active_tools(self, tool_names: list[str]) -> None:
        self._validate_tool_names(tool_names)
        previous_names = list(self._tools.keys())
        previous_active = list(self._active_tool_names)
        if self._phase != "idle":
            self._pending_mutations += 1
        self._active_tool_names = list(tool_names)
        await self._emit_own(
            ToolsUpdateEvent(
                type="tools_update",
                tool_names=list(self._tools.keys()),
                previous_tool_names=previous_names,
                active_tool_names=list(self._active_tool_names),
                previous_active_tool_names=previous_active,
                source="set",
            )
        )

    def get_resources(self) -> AgentHarnessResources:
        return self._resources.clone()

    async def set_resources(self, resources: AgentHarnessResources) -> None:
        self._assert_not_shut_down()
        previous_resources = self._resources.clone()
        self._resources = resources.clone()
        await self._emit_own(
            ResourcesUpdateEvent(
                type="resources_update",
                resources=self._resources.clone(),
                previous_resources=previous_resources,
            )
        )

    def get_stream_options(self) -> AgentHarnessStreamOptions:
        return self._stream_options.clone()

    async def set_stream_options(
        self,
        stream_options: AgentHarnessStreamOptions,
        patch: AgentHarnessStreamOptionsPatch | None = None,
    ) -> None:
        self._assert_not_shut_down()
        self._stream_options = apply_stream_options_patch(stream_options, patch)

    def get_steering_mode(self) -> QueueMode:
        return self._steering_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._assert_not_shut_down()
        self._steering_mode = mode

    def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._assert_not_shut_down()
        self._follow_up_mode = mode
