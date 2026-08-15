"""
AgentSession — coding-agent 中枢编排类

封装 pi_agent.Agent + 工具注入 + 会话持久化 + 事件转发。

用法:
    session = AgentSession(agent, session_manager, cwd, model)
    session.subscribe(lambda event: print(event))
    await session.prompt("read README.md")
    await session.wait_for_idle()
    await session.dispose()
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from pi_agent import (
    AfterToolCallContext,
    AfterToolCallResult,
    Agent,
    AgentEvent,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PythonExecutionEnv,
)
from pi_agent._agent import _default_convert_to_llm as _agent_default_convert_to_llm
from pi_agent.shell_output import execute_shell_with_capture
from pi_ai import AssistantMessage, DeferredHandle, Model, TextContent, Usage, UserMessage, now_ms
from pi_ai.api._shared import empty_usage
from pi_ai.types.common import ModelThinkingLevel, ThinkingLevel
from pi_ai.utils.estimate import calculate_context_tokens
from pi_ai.utils.overflow import is_context_overflow
from pi_ai.utils.retry import (
    RetryPolicy,
    compute_backoff_delay,
    is_retryable_error,
)

from ._session_manager_v4 import SessionManagerLike
from .frontmatter import strip_frontmatter
from .messages import convert_to_llm
from .model_resolver import ScopedModel
from .model_runtime import ModelRuntime
from pi_agent.prompt_templates import substitute_args

from .prompt_templates import PromptTemplateLoader, parse_command_args
from .skills import SkillLoader
from .model_utils import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    clamp_thinking_level,
    get_supported_thinking_levels,
    models_are_equal,
)
from .compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionPreparation,
    CompactionResult,
    CompactionSettings,
    compact,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from .cache_stats import compute_cache_waste
from .tools import create_all_tools


async def _record_operation(
    manager: SessionManagerLike,
    kind: str,
    **kwargs,
) -> str | None:
    """写入 operation record；管理器无此能力时静默跳过。"""
    start = getattr(manager, "start_operation", None)
    if start is None:
        return None
    try:
        return await start(kind, **kwargs)
    except Exception:
        return None


async def _finish_recorded_operation(
    manager: SessionManagerLike,
    run_id: str | None,
    outcome: str = "completed",
    error: dict[str, str] | None = None,
) -> None:
    """关闭 operation record；失败不阻断主流程。"""
    if run_id is None:
        return
    finish = getattr(manager, "finish_operation", None)
    if finish is None:
        return
    try:
        await finish(run_id, outcome=outcome, error=error)
    except Exception:
        pass


async def _record_usage_for_message(
    manager: SessionManagerLike,
    message: AgentMessage,
    entry_id: str,
) -> None:
    """把 assistant 消息的 usage 写入 usage record（无此能力时跳过）。"""
    record = getattr(manager, "record_usage", None)
    usage = message.get("usage")
    if record is None or not isinstance(usage, dict):
        return
    try:
        await record(
            cause="assistant",
            usage=usage,
            entry_id=entry_id,
            stop_reason=message.get("stopReason"),
        )
    except Exception:
        pass


@dataclass(slots=True)
class ModelCycleResult:
    """cycleModel 的结果。"""

    model: Model
    thinking_level: ThinkingLevel
    is_scoped: bool


@dataclass(slots=True)
class BashResult:
    """交互 bash 执行结果（对齐 TS BashResult）。"""

    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None


class AgentSession:
    """中枢会话对象 — 连接 Agent、工具、持久化、事件转发。

    支持扩展运行器（session 事件）、自动/手动压缩、分支摘要，
    以及重试（agent 内部 + turn 级）。
    """

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManagerLike,
        cwd: str,
        model: Model,
        *,
        tools_override: list[AgentTool] | None = None,
        # turn 级重试策略。None = 默认启用（enabled=True, max_retries=3）。
        # 传入 RetryPolicy(enabled=False) 可关闭。
        turn_retry_policy: RetryPolicy | None = None,
        # 自动压缩设置。None = 默认（enabled=True）。
        compaction_settings: CompactionSettings | None = None,
        # 模型运行时（Phase 1：setModel/cycleModel/可用模型列表）。
        model_runtime: ModelRuntime | None = None,
        # --models 循环列表（scope 模式优先于全量可用列表）。
        scoped_models: list[ScopedModel] | None = None,
        # Phase 4：技能 / 提示模板加载器（/skill:name 与 /templateName 展开）。
        skill_loader: SkillLoader | None = None,
        template_loader: PromptTemplateLoader | None = None,
        # Phase 5：扩展运行器（事件分发 / input 变换 / 注册项）。
        extension_runner=None,
        # 系统提示构建器（/reload 重建用）。
        system_prompt_builder: Callable[[], str] | None = None,
        # CLI/TUI 共享的扩展状态（system_prompt_builder 读取 active_tools）。
        extension_state: dict | None = None,
        # 未信任项目时拦截高风险工具（bash/write/edit；默认关闭，对齐 TS 不限制工具）。
        restrict_untrusted_tools: bool = False,
        # 会话启动事件附加元数据（对齐 TS SessionStartEvent）。
        session_start_event: dict | None = None,
        # 设置管理器（模型/思考级别/队列模式持久化，对齐 TS AgentSession）。
        settings_manager=None,
        # 工具允许/排除集合（同时作用于内置、SDK 自定义与扩展工具）。
        allowed_tool_names: set[str] | None = None,
        excluded_tool_names: set[str] | None = None,
    ):
        self._agent = agent
        # 编码代理使用完整消息转换器（bashExecution/compactionSummary/custom 等
        # 包装为 user 消息，对齐 TS coding-agent messages.ts）。显式传入的自定义
        # 转换器保持优先（AgentOptions.convert_to_llm 非默认值时不被覆盖）。
        if agent.convert_to_llm is _agent_default_convert_to_llm:
            agent.convert_to_llm = convert_to_llm
        self._session_manager = session_manager
        self._cwd = cwd
        self._model = model
        self._model_runtime = model_runtime
        self._scoped_models = list(scoped_models or [])
        self._skill_loader = skill_loader
        self._template_loader = template_loader
        self._extension_runner = extension_runner
        self._extension_tool_names: set[str] = set()
        self._system_prompt_builder = system_prompt_builder
        self._restrict_untrusted_tools = restrict_untrusted_tools
        self._settings_manager = settings_manager
        self._allowed_tool_names = allowed_tool_names
        self._excluded_tool_names = excluded_tool_names
        self._session_start_event = dict(session_start_event or {})
        if extension_runner is not None:
            extension_runner.bind_session(self)
        self._listeners: list[Callable[[dict[Any, Any]], None]] = []
        # CLI/TUI 共享的扩展状态（system_prompt_builder 用它读当前 runner）。
        self.extension_state: dict | None = extension_state
        # 项目信任状态（CLI/TUI 在启动解析后设置；None=未知）。
        self.project_trusted: bool | None = None
        self._after_response_tasks: set[asyncio.Task] = set()
        self._pending_writes: set[asyncio.Task[object]] = set()
        # None 对齐 TS 默认：turn 级重试开启（enabled=true, maxRetries=3）。
        self._turn_retry_policy = (
            turn_retry_policy if turn_retry_policy is not None else RetryPolicy()
        )
        self._compaction_settings = compaction_settings or DEFAULT_COMPACTION_SETTINGS
        # 溢出恢复已尝试标记（每次 prompt 重置；单次压缩+重试保护）。
        self._overflow_recovery_attempted = False
        # 压缩执行中标记（get_state / RPC 状态查询用）。
        self._is_compacting = False
        # turn 级重试退避期间的中止信号（None 表示空闲）。
        self._abort: asyncio.Event | None = None
        # turn 级耗时记录（get_session_stats 用）。
        self._turn_timings: list[dict] = []
        # 交互 bash（!/!!）运行中/待刷入上下文的状态。
        self._bash_abort_signals: set[asyncio.Event] = set()
        self._pending_bash_messages: list[dict[str, Any]] = []

        # 注入编码工具
        if tools_override is not None:
            tools = tools_override
        else:
            tools = create_all_tools(
                cwd,
                bash_session_env_provider=self._session_env_vars,
            )
        tools = self._filter_tools_by_name(tools)
        self._agent.state.tools = tools
        self._apply_extension_tools()

        # 恢复已有会话消息历史（打开已有会话时）
        existing_messages = self._session_manager.build_context()
        if existing_messages:
            self._agent.state.messages = existing_messages

        # 订阅 agent 事件 → 持久化 + 转发
        self._unsub_agent: Callable[[], None] | None = self._agent.subscribe(
            self._handle_agent_event
        )

        # before_provider_request / after_provider_response：包装 stream fn。
        try:
            base_stream_fn = self._agent._resolve_stream_fn()
        except RuntimeError:
            base_stream_fn = None
        if base_stream_fn is not None:
            self._agent.stream_function = self._wrap_stream_fn(base_stream_fn)

        # 扩展 tool_call / tool_result 事件：包装 agent 钩子（保留用户原有钩子）。
        self._agent.before_tool_call = self._wrap_before_tool_call(self._agent.before_tool_call)
        self._agent.after_tool_call = self._wrap_after_tool_call(self._agent.after_tool_call)

        # session_start（对齐 TS：会话开始事件）。
        self._emit_extension_event(
            "session_start",
            {
                "type": "session_start",
                "session_id": self.session_id,
                "cwd": self._cwd,
                "is_continuing": bool(existing_messages),
                **self._session_start_event,
            },
        )

    # ------------------------------------------------------------------
    # 扩展事件辅助
    # ------------------------------------------------------------------

    def _emit_extension_event(self, event_type: str, data: dict) -> None:
        """把会话级事件异步转发给扩展（无 handler 或无线程时不派发）。"""
        runner = self._extension_runner
        if runner is None or not runner.has_handlers(event_type):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        runner._schedule(runner.emit_event(event_type, data))

    def _wrap_before_tool_call(self, base):
        """包装 before_tool_call：先派发扩展 tool_call 事件（可 block / 改参数）。"""

        async def _before(ctx: BeforeToolCallContext):
            if self._restrict_untrusted_tools and self.project_trusted is not True:
                tool_name = ctx.tool_call.get("name", "")
                if tool_name in ("bash", "write", "edit"):
                    return BeforeToolCallResult(
                        block=True,
                        reason="Project is not trusted; high-risk tools are disabled",
                    )
            runner = self._extension_runner
            if runner is not None and runner.has_handlers("tool_call"):
                results = await runner.emit_event(
                    "tool_call",
                    {
                        "type": "tool_call",
                        "toolCallId": ctx.tool_call["id"],
                        "toolName": ctx.tool_call["name"],
                        "input": ctx.args,
                    },
                )
                for result in reversed(results):
                    if not isinstance(result, dict):
                        continue
                    if result.get("block"):
                        return BeforeToolCallResult(
                            block=True, reason=result.get("reason") or "Blocked by extension"
                        )
                    if "input" in result:
                        ctx.args = result["input"]
                        break
            if base is not None:
                return base(ctx)
            return None

        return _before

    def _wrap_after_tool_call(self, base):
        """包装 after_tool_call：先派发扩展 tool_result 事件（可覆盖结果）。"""

        async def _after(ctx: AfterToolCallContext):
            runner = self._extension_runner
            if runner is not None and runner.has_handlers("tool_result"):
                results = await runner.emit_event(
                    "tool_result",
                    {
                        "type": "tool_result",
                        "toolCallId": ctx.tool_call["id"],
                        "toolName": ctx.tool_call["name"],
                        "input": ctx.args,
                        "result": ctx.result,
                        "isError": ctx.is_error,
                    },
                )
                for result in reversed(results):
                    if not isinstance(result, dict):
                        continue
                    if any(
                        key in result
                        for key in ("content", "details", "is_error", "usage", "terminate")
                    ):
                        return AfterToolCallResult(
                            content=result.get("content"),
                            details=result.get("details"),
                            is_error=result.get("is_error"),
                            usage=result.get("usage"),
                            terminate=result.get("terminate"),
                        )
            if base is not None:
                return base(ctx)
            return None

        return _after

    def _wrap_stream_fn(self, base):
        """包装 stream_fn：派发 before_provider_request / after_provider_response。"""

        async def _wrapped(model, context, options=None):
            runner = self._extension_runner
            stream_options = dict(options or {})
            if runner is not None and runner.has_handlers("context"):
                results = await runner.emit_event(
                    "context",
                    {
                        "type": "context",
                        "messages": copy.deepcopy(context.messages or []),
                    },
                )
                for result in reversed(results):
                    if isinstance(result, dict) and isinstance(result.get("messages"), list):
                        context.messages = result["messages"]
            if runner is not None and runner.has_handlers("before_provider_headers"):
                results = await runner.emit_event(
                    "before_provider_headers",
                    {
                        "type": "before_provider_headers",
                        "model": model,
                        "session_id": self.session_id,
                        "headers": stream_options.get("headers") or {},
                    },
                )
                for result in reversed(results):
                    if isinstance(result, dict) and isinstance(result.get("headers"), dict):
                        merged = dict(stream_options.get("headers") or {})
                        merged.update(result["headers"])
                        stream_options["headers"] = merged
            if runner is not None and runner.has_handlers("before_provider_request"):
                results = await runner.emit_event(
                    "before_provider_request",
                    {
                        "type": "before_provider_request",
                        "model": model,
                        "session_id": self.session_id,
                        "stream_options": stream_options,
                    },
                )
                for result in reversed(results):
                    if isinstance(result, dict) and isinstance(result.get("stream_options"), dict):
                        stream_options.update(result["stream_options"])

            result = base(model, context, stream_options)
            if inspect.isawaitable(result):
                result = await result
            stream = result

            if runner is not None and runner.has_handlers("after_provider_response"):
                emitted = {"done": False}

                async def _wait_and_emit() -> None:
                    try:
                        message = await stream.result()
                    except Exception:
                        return
                    if emitted["done"]:
                        return
                    emitted["done"] = True
                    await runner.emit_event(
                        "after_provider_response",
                        {
                            "type": "after_provider_response",
                            "model": model,
                            "session_id": self.session_id,
                            "response": message,
                        },
                    )

                task = asyncio.create_task(_wait_and_emit())
                self._after_response_tasks.add(task)
                task.add_done_callback(self._after_response_tasks.discard)
            return stream

        return _wrapped

    async def _run_compaction_hooks(
        self,
        preparation: CompactionPreparation,
        reason: str,
        will_retry: bool,
        custom_instructions: str | None,
    ) -> tuple[bool, CompactionResult | None, bool]:
        """session_before_compact 事件：取消 / 扩展提供自定义压缩结果。

        返回 (cancelled, result, from_extension)。
        """
        runner = self._extension_runner
        if runner is None or not runner.has_handlers("session_before_compact"):
            return False, None, False
        results = await runner.emit_event(
            "session_before_compact",
            {
                "type": "session_before_compact",
                "preparation": preparation,
                "branchEntries": self._session_manager.get_branch(),
                "customInstructions": custom_instructions,
                "reason": reason,
                "willRetry": will_retry,
            },
        )
        for result in reversed(results):
            if not isinstance(result, dict):
                continue
            if result.get("cancel"):
                return True, None, False
            compaction = result.get("compaction")
            if not isinstance(compaction, dict):
                continue
            return (
                False,
                CompactionResult(
                    summary=compaction.get("summary", ""),
                    first_kept_entry_id=(
                        compaction.get("firstKeptEntryId") or preparation.first_kept_entry_id
                    ),
                    tokens_before=(
                        int(compaction["tokensBefore"])
                        if compaction.get("tokensBefore") is not None
                        else preparation.tokens_before
                    ),
                    usage=compaction.get("usage") or empty_usage(),
                    details=compaction.get("details") or {},
                ),
                True,
            )
        return False, None, False

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_manager(self) -> SessionManagerLike:
        return self._session_manager

    @property
    def skill_loader(self) -> SkillLoader | None:
        return self._skill_loader

    @property
    def template_loader(self) -> PromptTemplateLoader | None:
        return self._template_loader

    @property
    def extension_runner(self):
        return self._extension_runner

    @property
    def session_id(self) -> str:
        return self._session_manager.session_id

    @property
    def session_file(self) -> str | None:
        """已持久化会话的文件路径；内存会话为 None。"""
        if not self._session_manager.is_persisted():
            return None
        path = self._session_manager.session_path
        return str(path) if path is not None else None

    @property
    def session_name(self) -> str | None:
        return self._session_manager.session_name

    def set_session_name(self, name: str) -> None:
        previous = self._session_manager.session_name
        self._session_manager.set_session_name(name)
        self._emit_extension_event(
            "session_info_changed",
            {
                "type": "session_info_changed",
                "name": name,
                "previousName": previous,
            },
        )

    def _session_env_vars(self) -> dict[str, str]:
        """bash 工具子进程的会话环境变量（对齐 TS bash 会话注入）。"""
        env: dict[str, str] = {
            "PI_SESSION_ID": self.session_id,
        }
        if self.session_file is not None:
            env["PI_SESSION_FILE"] = self.session_file
        if self._model is not None:
            env["PI_PROVIDER"] = self._model.provider
            env["PI_MODEL"] = self._model.id
        env["PI_REASONING_LEVEL"] = str(self.thinking_level)
        return env

    @property
    def thinking_level(self) -> ThinkingLevel:
        return cast(ThinkingLevel, self._agent.state.thinking_level)

    @property
    def is_streaming(self) -> bool:
        return bool(self._agent.state.is_streaming)

    @property
    def is_compacting(self) -> bool:
        return self._is_compacting

    @property
    def pending_message_count(self) -> int:
        return int(self._agent.pending_message_count)

    @property
    def turn_timings(self) -> list[dict]:
        """每轮耗时记录（[{startedAtMs, durationMs}]）。"""
        return list(self._turn_timings)

    @property
    def steering_mode(self):
        return self._agent.steering_mode

    @steering_mode.setter
    def steering_mode(self, mode) -> None:
        self._agent.steering_mode = mode
        if self._settings_manager is not None:
            setter = getattr(self._settings_manager, "set_steering_mode", None)
            if setter is not None:
                setter(mode)

    @property
    def follow_up_mode(self):
        return self._agent.follow_up_mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode) -> None:
        self._agent.follow_up_mode = mode
        if self._settings_manager is not None:
            setter = getattr(self._settings_manager, "set_follow_up_mode", None)
            if setter is not None:
                setter(mode)

    @property
    def auto_compaction_enabled(self) -> bool:
        return bool(self._compaction_settings.enabled)

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        self._compaction_settings.enabled = enabled

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        current = self._turn_retry_policy
        if current is None:
            self._turn_retry_policy = RetryPolicy(
                enabled=enabled,
                max_retries=3,
            )
            return
        self._turn_retry_policy = RetryPolicy(
            enabled=enabled,
            max_retries=current.max_retries,
            base_delay_ms=current.base_delay_ms,
            max_delay_ms=current.max_delay_ms,
            jitter=current.jitter,
        )

    def abort_retry(self) -> None:
        """中止 turn 级重试的退避等待（不影响当前运行）。"""
        if self._abort is not None:
            self._abort.set()

    def get_available_models(self) -> list[Model]:
        """获取可循环的模型列表（scope 优先，否则运行时可用快照）。"""
        if self._scoped_models:
            return [scoped.model for scoped in self._scoped_models]
        if self._model_runtime is not None:
            return self._model_runtime.get_available_snapshot()
        return []

    @property
    def scoped_models(self) -> list[ScopedModel]:
        return list(self._scoped_models)

    def set_scoped_models(self, scoped_models: list[ScopedModel]) -> None:
        """设置 Ctrl+P 循环的 scope 列表（空 = 全部可用）。"""
        self._scoped_models = list(scoped_models)

    def set_extension_runner(self, runner) -> None:
        """替换扩展运行器并立即绑定当前会话（/reload 用）。"""
        self._extension_runner = runner
        if runner is not None:
            runner.bind_session(self)
        self._apply_extension_tools()

    def _tool_is_allowed(self, name: str) -> bool:
        """按 TS allowedToolNames/excludedToolNames 过滤（None 表示不限制）。"""
        if self._allowed_tool_names is not None and name not in self._allowed_tool_names:
            return False
        if self._excluded_tool_names is not None and name in self._excluded_tool_names:
            return False
        return True

    def _filter_tools_by_name(self, tools: list[Any]) -> list[Any]:
        return [tool for tool in tools if self._tool_is_allowed(tool.name)]

    def _apply_extension_tools(self) -> None:
        """把扩展注册的工具合并进 agent 工具集（同名覆盖内置，结果归一化为 AgentToolResult）。

        扩展工具同样遵守 allowed/excluded 工具集合（对齐 TS _refreshToolRegistry）。
        """
        runner = self._extension_runner
        if runner is None:
            return
        definitions = [
            definition
            for definition in runner.get_registered_tools()
            if self._tool_is_allowed(definition.name)
        ]
        current = list(self._agent.state.tools or [])
        # 移除上一轮已应用的扩展工具（同名由新 runner 覆盖）。
        current = [tool for tool in current if tool.name not in self._extension_tool_names]
        if not definitions:
            self._agent.state.tools = current
            self._extension_tool_names = set()
            if self.extension_state is not None:
                self.extension_state["active_tools"] = list(current)
            if self._system_prompt_builder is not None:
                self.rebuild_system_prompt()
            return

        from pi_agent import AgentToolResult

        extension_tools: dict[str, Any] = {}
        for definition in definitions:
            original = definition.execute

            async def execute(
                tool_call_id,
                params,
                signal=None,
                on_update=None,
                context=None,
                _original=original,
            ):
                raw = _original(tool_call_id, params, signal, on_update, context)
                if inspect.isawaitable(raw):
                    raw = await raw
                if isinstance(raw, AgentToolResult):
                    return raw
                if isinstance(raw, dict):
                    return AgentToolResult(
                        content=raw.get("content") or [],
                        details=raw.get("details"),
                        terminate=raw.get("terminate"),
                        usage=raw.get("usage"),
                    )
                if raw is None:
                    return AgentToolResult(content=[])
                # 字符串/标量返回归一化为文本内容（对齐 TS），
                # 否则 agent 循环的 _make_tool_result_message 会因缺 content 崩溃。
                return AgentToolResult(content=[{"type": "text", "text": str(raw)}])

            extension_tools[definition.name] = AgentTool(
                name=definition.name,
                label=definition.label or definition.name,
                description=definition.description,
                input_schema=definition.parameters or {"type": "object", "properties": {}},
                prompt_snippet=definition.prompt_snippet or None,
                prompt_guidelines=definition.prompt_guidelines,
                execution_mode=definition.execution_mode,
                execute=execute,
            )

        merged = [tool for tool in current if tool.name not in extension_tools]
        merged.extend(extension_tools.values())
        self._agent.state.tools = merged
        self._extension_tool_names = set(extension_tools)
        if self.extension_state is not None:
            self.extension_state["active_tools"] = list(merged)
        if self._system_prompt_builder is not None:
            self.rebuild_system_prompt()

    def rebuild_system_prompt(self) -> str | None:
        """重建系统提示（上下文文件 / 技能变化后调用，/reload 用）。"""
        if self._system_prompt_builder is None:
            return None
        prompt = self._system_prompt_builder()
        self._agent.state.system_prompt = prompt
        return prompt

    def steer(self, text: str, images: list | None = None) -> None:
        """入队一条 steering 消息（Agent 运行中注入；images 附到消息内容）。"""
        content: list = [TextContent(type="text", text=text)]
        if images:
            content.extend(images)
        self._agent.steer(UserMessage(role="user", content=content, timestamp=now_ms()))

    def follow_up(self, text: str, images: list | None = None) -> None:
        """入队一条 follow-up 消息（Agent 即将停止时继续）。"""
        content: list = [TextContent(type="text", text=text)]
        if images:
            content.extend(images)
        self._agent.follow_up(UserMessage(role="user", content=content, timestamp=now_ms()))

    def get_last_assistant_text(self) -> str | None:
        """返回最后一条 assistant 消息的纯文本。"""
        for message in reversed(self.get_messages()):
            if message.get("role") != "assistant":
                continue
            parts = [
                str(block.get("text", ""))
                for block in cast(dict[str, Any], message).get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "".join(parts)
            return text or None
        return None

    def get_context_usage(self) -> dict | None:
        """估算当前上下文 token 用量（对齐 TS getContextUsage）。"""
        from pi_agent.compaction_utils import estimate_context_tokens

        estimate = estimate_context_tokens(self._agent.state.messages)
        if estimate.last_usage_index is None:
            return None
        return {"tokens": estimate.tokens}

    def get_session_stats(self) -> dict:
        """汇总会话统计（对齐 TS SessionStats）。"""
        messages = self.get_messages()
        user_messages = 0
        assistant_messages = 0
        tool_calls = 0
        tool_results = 0
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0}
        cost = 0.0
        for message in messages:
            role = message.get("role")
            if role == "user":
                user_messages += 1
            elif role == "assistant":
                assistant_messages += 1
                for block in cast(dict[str, Any], message).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "toolCall":
                        tool_calls += 1
                usage = message.get("usage")
                if isinstance(usage, dict):
                    for key in ("input", "output", "cache_read", "cache_write", "total_tokens"):
                        tokens[key.replace("total_tokens", "total")] += usage.get(key, 0) or 0
                    cost += (usage.get("cost") or {}).get("total", 0) or 0
            elif role == "toolResult":
                tool_results += 1

        turn_count = len(self._turn_timings)
        total_turn_ms = sum(timing.get("durationMs", 0) for timing in self._turn_timings)
        cache_stats = compute_cache_waste(
            [cast(dict[Any, Any], message) for message in messages],
            self._model_runtime,
        )
        return {
            "sessionFile": self.session_file,
            "sessionId": self.session_id,
            "userMessages": user_messages,
            "assistantMessages": assistant_messages,
            "toolCalls": tool_calls,
            "toolResults": tool_results,
            "totalMessages": len(messages),
            "tokens": tokens,
            "cost": cost,
            "contextUsage": self.get_context_usage(),
            "turnTimings": {
                "turnCount": turn_count,
                "totalMs": int(total_turn_ms),
                "averageMs": int(total_turn_ms / turn_count) if turn_count else 0,
                "lastMs": int(self._turn_timings[-1]["durationMs"]) if turn_count else 0,
            },
            "cacheStats": cache_stats,
        }

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult | None:
        """手动压缩（RPC compact 命令）。

        对齐 TS compact()：压缩前中止当前运行；custom_instructions 会写入
        摘要请求的自定义指令段。
        """
        if self._model is None:
            return None
        # 对齐 TS compact()：压缩前先中止当前 agent 运行。
        await self.abort()
        entries = self._session_manager.get_entries()
        context_messages = self._agent.state.messages
        preparation: CompactionPreparation | None = prepare_compaction(
            entries, context_messages, self._compaction_settings
        )
        if preparation is None:
            return None

        self._emit({"type": "compaction_start", "reason": "manual"})
        self._is_compacting = True
        run_id = await _record_operation(
            self._session_manager,
            "compaction",
            custom_instructions=custom_instructions,
        )
        outcome = "completed"
        error: dict[str, str] | None = None
        try:
            cancelled, ext_result, from_extension = await self._run_compaction_hooks(
                preparation, "manual", False, custom_instructions
            )
            if cancelled:
                self._emit(
                    {
                        "type": "compaction_end",
                        "reason": "manual",
                        "result": None,
                        "aborted": True,
                        "willRetry": False,
                    }
                )
                return None
            if ext_result is None:
                summarization_auth = await self._get_summarization_request_auth(self._model)
                result = await compact(
                    preparation,
                    summarization_auth["model"],
                    api_key=summarization_auth.get("apiKey"),
                    headers=summarization_auth.get("headers"),
                    env=summarization_auth.get("env"),
                    stream_fn=self._agent._resolve_stream_fn(),
                    custom_instructions=custom_instructions,
                    thinking_level=self._agent.state.thinking_level,
                )
            else:
                result = ext_result
            if isinstance(getattr(result, "usage", None), dict):
                record = getattr(self._session_manager, "record_usage", None)
                if record is not None:
                    try:
                        await record(
                            cause="compaction",
                            usage=cast(dict, result.usage),
                            run_id=run_id,
                        )
                    except Exception:
                        pass
            entry_id = await self._session_manager.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                result.details,
            )
            self._agent.state.messages = self._session_manager.build_context()
            self._emit_extension_event(
                "session_compact",
                {
                    "type": "session_compact",
                    "compactionEntry": {
                        "id": entry_id,
                        "type": "compaction",
                        "summary": result.summary,
                        "firstKeptEntryId": result.first_kept_entry_id,
                        "tokensBefore": result.tokens_before,
                    },
                    "fromExtension": from_extension,
                    "reason": "manual",
                    "willRetry": False,
                },
            )
            self._emit(
                {
                    "type": "compaction_end",
                    "reason": "manual",
                    "result": result,
                    "aborted": False,
                    "willRetry": False,
                }
            )
            return result
        except Exception as exc:
            outcome = "failed"
            error = {"code": "error", "message": str(exc)}
            self._emit(
                {
                    "type": "compaction_end",
                    "reason": "manual",
                    "result": None,
                    "aborted": False,
                    "willRetry": False,
                    "error": str(exc),
                }
            )
            return None
        finally:
            await _finish_recorded_operation(self._session_manager, run_id, outcome, error)
            self._is_compacting = False

    async def _get_summarization_request_auth(self, model: Model) -> dict:
        """解析摘要请求级认证（对齐 TS _getSummarizationRequestAuth）。

        摘要与主请求共用同一 stream_fn 时会自动继承认证；这里额外支持
        显式覆盖：apiKey / headers / env / baseUrl（auth 解析失败时静默回退）。
        """
        runtime = self._model_runtime
        if runtime is None:
            return {"model": model}
        try:
            result = await runtime.get_auth(model)
        except Exception:
            return {"model": model}
        if result is None:
            return {"model": model}
        auth = getattr(result, "auth", None) or {}
        request_model = model
        base_url = auth.get("base_url")
        if base_url:
            request_model = replace(model, base_url=base_url)
        headers = auth.get("headers") or {}
        return {
            "model": request_model,
            "apiKey": auth.get("api_key"),
            "headers": {k: v for k, v in headers.items() if v is not None} or None,
            "env": getattr(result, "env", None),
        }

    async def navigate_to(
        self,
        entry_id: str,
        *,
        summarize: bool = True,
        custom_instructions: str | None = None,
    ) -> bool:
        """导航到目标条目：生成分支摘要（可选）→ 移动 leaf → 重建上下文。"""
        manager = self._session_manager
        old_leaf = manager.get_leaf_id()
        if old_leaf == entry_id:
            return False
        if manager.get_entry(entry_id) is None:
            raise ValueError(f"Entry not found: {entry_id}")

        run_id = await _record_operation(
            self._session_manager,
            "navigation",
            target_id=entry_id,
            summarize=summarize,
            custom_instructions=custom_instructions,
        )
        outcome = "completed"
        error: dict[str, str] | None = None
        try:
            return await self._navigate_to_impl(
                entry_id,
                summarize=summarize,
                custom_instructions=custom_instructions,
                old_leaf=old_leaf,
                manager=manager,
            )
        except BaseException as exc:
            outcome = "failed"
            error = {"code": "error", "message": str(exc)}
            raise
        finally:
            await _finish_recorded_operation(self._session_manager, run_id, outcome, error)

    async def _navigate_to_impl(
        self,
        entry_id: str,
        *,
        summarize: bool,
        custom_instructions: str | None,
        old_leaf: str | None,
        manager: SessionManagerLike,
    ) -> bool:
        """navigate_to 主体（由 operation record 包装调用）。"""
        summary: dict | None = None
        from_extension = False
        runner = self._extension_runner
        if runner is not None and runner.has_handlers("session_before_tree"):
            results = await runner.emit_event(
                "session_before_tree",
                {
                    "type": "session_before_tree",
                    "targetId": entry_id,
                    "oldLeafId": old_leaf,
                },
            )
            for result in reversed(results):
                if not isinstance(result, dict):
                    continue
                if result.get("cancel"):
                    return False
                if isinstance(result.get("summary"), dict):
                    summary = result["summary"]
                    from_extension = True
                    break

        summary_model = self._model
        if summary is None and summarize and old_leaf is not None and summary_model is not None:
            from pi_agent.branch_summarization import (
                collect_entries_for_branch_summary,
                generate_branch_summary,
            )

            class _BranchSessionAdapter:
                """branch_summarization 需要 async get_branch/get_entry。"""

                def __init__(self, manager: SessionManagerLike) -> None:
                    self._manager = manager

                async def get_branch(self, from_id: str | None = None):
                    return self._manager.get_branch(from_id)

                async def get_entry(self, entry_id: str):
                    return self._manager.get_entry(entry_id)

            try:
                collected = await collect_entries_for_branch_summary(
                    _BranchSessionAdapter(manager), old_leaf, entry_id
                )
                ok, result = await generate_branch_summary(
                    collected["entries"],
                    stream_fn=self._agent._resolve_stream_fn(),
                    model=summary_model,
                    custom_instructions=custom_instructions,
                )
            except Exception as exc:
                ok, result = False, exc
            if ok and isinstance(result, dict):
                summary = {
                    "summary": result.get("summary", ""),
                    "details": result,
                    "fromHook": True,
                }
                if isinstance(result.get("usage"), dict):
                    record = getattr(manager, "record_usage", None)
                    if record is not None:
                        try:
                            await record(
                                cause="branch_summary",
                                usage=cast(dict, result["usage"]),
                            )
                        except Exception:
                            pass

        await manager.move_to(entry_id, summary)
        self._agent.state.messages = manager.build_context()
        self._emit_extension_event(
            "session_tree",
            {
                "type": "session_tree",
                "newLeafId": entry_id,
                "oldLeafId": old_leaf,
                "summaryEntry": summary,
                "fromExtension": from_extension,
            },
        )
        self._emit(
            {
                "type": "navigated",
                "entryId": entry_id,
                "fromEntryId": old_leaf,
            }
        )
        return True

    def _persist_default_model(self, model: Model) -> None:
        """切换模型时同步保存 settings 默认值（对齐 TS setDefaultModelAndProvider）。"""
        if self._settings_manager is not None:
            setter = getattr(self._settings_manager, "set_default_model_and_provider", None)
            if setter is not None:
                setter(model.provider, model.id)

    async def set_model(self, model: Model) -> None:
        """切换模型：校验认证 → 更新 agent state → 记录会话 → 重算思考级别。"""
        runtime = self._model_runtime
        if runtime is not None:
            check = await runtime.check_auth(model.provider)
            if check is None:
                raise RuntimeError(f"No API key for {model.provider}/{model.id}")

        previous_model = self._model
        thinking_level = self._get_thinking_level_for_model_switch()
        self._agent.state.model = model
        await self._session_manager.append_model_change(model.provider, model.id)
        self._persist_default_model(model)
        self._model = model
        self.set_thinking_level(thinking_level)
        if not models_are_equal(previous_model, model):
            self._emit(
                {
                    "type": "model_changed",
                    "model": model,
                    "previousModel": previous_model,
                }
            )
            self._emit_extension_event(
                "model_select",
                {
                    "type": "model_select",
                    "model": model,
                    "previousModel": previous_model,
                    "provider": model.provider,
                    "modelId": model.id,
                },
            )

    async def cycle_model(self, direction: int = 1) -> ModelCycleResult | None:
        """循环切换模型（正数向前 / 负数向后）。"""
        if self._scoped_models:
            return await self._cycle_scoped_model(direction)
        return await self._cycle_available_model(direction)

    async def _cycle_scoped_model(self, direction: int) -> ModelCycleResult | None:
        runtime = self._model_runtime
        if runtime is None:
            return None
        checks = await asyncio.gather(
            *(runtime.check_auth(scoped.model.provider) for scoped in self._scoped_models)
        )
        scoped = [
            scoped
            for scoped, check in zip(self._scoped_models, checks, strict=True)
            if check is not None
        ]
        if len(scoped) <= 1:
            return None

        current_index = next(
            (
                index
                for index, entry in enumerate(scoped)
                if models_are_equal(entry.model, self._model)
            ),
            -1,
        )
        if current_index == -1:
            current_index = 0
        length = len(scoped)
        next_index = (
            (current_index + 1) % length
            if direction >= 0
            else (current_index - 1 + length) % length
        )
        next_scoped = scoped[next_index]
        thinking_level = self._get_thinking_level_for_model_switch(next_scoped.thinking_level)
        previous_model = self._model
        self._agent.state.model = next_scoped.model
        await self._session_manager.append_model_change(
            next_scoped.model.provider, next_scoped.model.id
        )
        self._persist_default_model(next_scoped.model)
        self._model = next_scoped.model
        self.set_thinking_level(thinking_level)
        if not models_are_equal(previous_model, next_scoped.model):
            self._emit(
                {
                    "type": "model_changed",
                    "model": next_scoped.model,
                    "previousModel": previous_model,
                }
            )
            self._emit_extension_event(
                "model_select",
                {
                    "type": "model_select",
                    "model": next_scoped.model,
                    "previousModel": previous_model,
                    "provider": next_scoped.model.provider,
                    "modelId": next_scoped.model.id,
                },
            )
        return ModelCycleResult(next_scoped.model, self.thinking_level, True)

    async def _cycle_available_model(self, direction: int) -> ModelCycleResult | None:
        runtime = self._model_runtime
        if runtime is None:
            return None
        available = await runtime.get_available()
        if len(available) <= 1:
            return None

        current_index = next(
            (
                index
                for index, model in enumerate(available)
                if models_are_equal(model, self._model)
            ),
            -1,
        )
        if current_index == -1:
            current_index = 0
        length = len(available)
        next_index = (
            (current_index + 1) % length
            if direction >= 0
            else (current_index - 1 + length) % length
        )
        next_model = available[next_index]
        thinking_level = self._get_thinking_level_for_model_switch()
        previous_model = self._model
        self._agent.state.model = next_model
        await self._session_manager.append_model_change(next_model.provider, next_model.id)
        self._persist_default_model(next_model)
        self._model = next_model
        self.set_thinking_level(thinking_level)
        if not models_are_equal(previous_model, next_model):
            self._emit(
                {
                    "type": "model_changed",
                    "model": next_model,
                    "previousModel": previous_model,
                }
            )
            self._emit_extension_event(
                "model_select",
                {
                    "type": "model_select",
                    "model": next_model,
                    "previousModel": previous_model,
                    "provider": next_model.provider,
                    "modelId": next_model.id,
                },
            )
        return ModelCycleResult(next_model, self.thinking_level, False)

    # ------------------------------------------------------------------
    # 思考级别管理
    # ------------------------------------------------------------------

    def get_available_thinking_levels(self) -> list[ModelThinkingLevel]:
        if self._model is None:
            return list(THINKING_LEVELS)
        return get_supported_thinking_levels(self._model)

    def supports_thinking(self) -> bool:
        return bool(self._model is not None and self._model.reasoning)

    def set_thinking_level(self, level: ModelThinkingLevel) -> None:
        """设置思考级别（按模型能力收敛；仅在变化时持久化）。"""
        available = self.get_available_thinking_levels()
        effective = level if level in available else self._clamp_thinking_level(level, available)
        previous = self._agent.state.thinking_level
        if effective == previous:
            return
        self._agent.state.thinking_level = effective
        self._persist_thinking_level_change(effective)
        if self._settings_manager is not None:
            setter = getattr(self._settings_manager, "set_default_thinking_level", None)
            if setter is not None and (self.supports_thinking() or effective != "off"):
                setter(effective)
        self._emit(
            {
                "type": "thinking_level_changed",
                "level": effective,
                "previousLevel": previous,
            }
        )
        self._emit_extension_event(
            "thinking_level_select",
            {
                "type": "thinking_level_select",
                "level": effective,
                "previousLevel": previous,
            },
        )

    def cycle_thinking_level(self) -> ModelThinkingLevel | None:
        if not self.supports_thinking():
            return None
        levels = self.get_available_thinking_levels()
        current_index = levels.index(self.thinking_level) if self.thinking_level in levels else -1
        next_index = (current_index + 1) % len(levels)
        next_level = levels[next_index]
        self.set_thinking_level(next_level)
        return next_level

    def _get_thinking_level_for_model_switch(
        self, explicit_level: ModelThinkingLevel | None = None
    ) -> ThinkingLevel:
        if explicit_level is not None:
            return cast(ThinkingLevel, explicit_level)
        if not self.supports_thinking():
            if self._settings_manager is not None:
                saved = self._settings_manager.get_default_thinking_level()
                return cast(ThinkingLevel, saved or DEFAULT_THINKING_LEVEL)
            return DEFAULT_THINKING_LEVEL
        return self.thinking_level

    def _clamp_thinking_level(
        self, level: ModelThinkingLevel, _available_levels: list[ModelThinkingLevel]
    ) -> ThinkingLevel:
        if self._model is None:
            return "off"
        return cast(ThinkingLevel, clamp_thinking_level(self._model, level))

    def _persist_thinking_level_change(self, level: ModelThinkingLevel) -> None:
        """后台持久化思考级别变更（仅在运行中的事件循环内执行）。

        任务纳入 _pending_writes，dispose 时等待写入完成。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        task = asyncio.create_task(self._session_manager.append_thinking_level_change(level))
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

    async def prompt(
        self,
        text: str,
        images: list | None = None,
        *,
        preflight_result: Callable[[bool], Any] | None = None,
        streaming_behavior: str | None = None,
        source: str = "interactive",
        expand_prompt_templates: bool = True,
    ) -> None:
        """发送用户消息，触发完整的 Agent 循环 + 工具执行。

        对齐 TS AgentSession.prompt：
        - 扩展命令在发送前执行（即使正在 streaming）；
        - input 扩展事件可拦截/变换；
        - 正在 streaming 时必须提供 streamingBehavior（steer/followUp）；
        - 非 streaming 时校验 model 与认证后才回调 preflightResult(true)。
        """
        self._abort = asyncio.Event()
        started = time.perf_counter()
        started_at_ms = int(time.time() * 1000)
        run_id = await _record_operation(
            self._session_manager,
            "run",
            original_prompt=[
                {
                    "role": "user",
                    "content": text,
                    "timestamp": int(time.time() * 1000),
                }
            ],
        )
        outcome = "completed"
        error: dict[str, str] | None = None
        preflight_ok = False
        try:
            self._overflow_recovery_attempted = False

            # 1. 扩展命令（/command）优先执行，不进入 LLM。
            if expand_prompt_templates and text.startswith("/"):
                if await self._try_execute_extension_command(text):
                    preflight_ok = True
                    if preflight_result is not None:
                        preflight_result(True)
                    return

            # 2. input 事件（扩展可拦截/变换）。
            current_text = text
            current_images = images
            if self._extension_runner is not None:
                current_text, action = await self._extension_runner.emit_input(
                    current_text,
                    images=current_images,
                    source=source,
                    streaming_behavior=streaming_behavior,
                )
                if action == "handled":
                    preflight_ok = True
                    if preflight_result is not None:
                        preflight_result(True)
                    return

            # 3. 技能与提示模板展开。
            expanded = current_text
            if expand_prompt_templates:
                expanded = self.expand_prompt(expanded)
            if current_text.startswith("/skill:") and expanded != current_text:
                skill_name = current_text[7:].split(" ", 1)[0] if len(current_text) > 7 else ""
                self._emit({"type": "skill_invocation", "skill": skill_name})

            # 4. Streaming 时按 streamingBehavior 入队。
            if self.is_streaming:
                if streaming_behavior not in ("steer", "followUp", "follow_up"):
                    raise RuntimeError(
                        "Agent is already processing. Specify streamingBehavior "
                        "('steer' or 'followUp') to queue the message."
                    )
                if streaming_behavior in ("followUp", "follow_up"):
                    self.follow_up(expanded, current_images)
                else:
                    self.steer(expanded, current_images)
                preflight_ok = True
                if preflight_result is not None:
                    preflight_result(True)
                return

            # 5. 发送前 flush 延迟 bash 消息。
            self._flush_pending_bash_messages()

            # 6. 校验 model 与认证（对齐 TS preflight）。
            if self._model is None:
                raise RuntimeError("No model selected.")
            if self._model_runtime is not None:
                has_auth = self._model_runtime.has_configured_auth(self._model.provider)
                if not has_auth:
                    checked = await self._model_runtime.check_auth(self._model.provider)
                    has_auth = checked is not None
                if not has_auth:
                    provider = self._model.provider
                    if self._model_runtime.is_using_oauth(provider):
                        raise RuntimeError(
                            f'Authentication failed for "{provider}". Credentials may have expired '
                            f"or network is unavailable. Run '/login {provider}' to re-authenticate."
                        )
                    raise RuntimeError(f"No API key configured for provider '{provider}'.")

            # 7. 发送前检查上一轮 aborted/error 响应是否需要压缩。
            if self._last_assistant_message() is not None:
                await self._check_compaction(skip_aborted_check=False)

            # 8. before_agent_start（扩展可覆盖系统提示 / 本轮 prompt / 注入消息）。
            injected_messages: list[dict] = []
            if self._extension_runner is not None and self._extension_runner.has_handlers(
                "before_agent_start"
            ):
                results = await self._extension_runner.emit_event(
                    "before_agent_start",
                    {
                        "type": "before_agent_start",
                        "prompt": expanded,
                        "system_prompt": self._agent.state.system_prompt,
                    },
                )
                original_system_prompt = self._agent.state.system_prompt
                try:
                    for result in reversed(results):
                        if not isinstance(result, dict):
                            continue
                        if isinstance(result.get("system_prompt"), str):
                            self._agent.state.system_prompt = result["system_prompt"]
                        if isinstance(result.get("prompt"), str):
                            expanded = result["prompt"]
                        for message in result.get("messages") or []:
                            if isinstance(message, dict) and "content" in message:
                                injected_messages.append(
                                    {
                                        "role": message.get("role", "custom"),
                                        "customType": message.get("customType", "extension"),
                                        "content": message.get("content"),
                                        "display": message.get("display", False),
                                        "details": message.get("details"),
                                        "timestamp": now_ms(),
                                    }
                                )
                        # 兼容旧版扩展单 message 返回。
                        message = result.get("message")
                        if isinstance(message, dict) and isinstance(message.get("content"), str):
                            injected_messages.append(
                                {
                                    "role": "user",
                                    "content": message["content"],
                                    "customType": message.get("customType", "extension"),
                                    "display": message.get("display", False),
                                    "timestamp": now_ms(),
                                }
                            )
                        if result:
                            break
                    messages_to_send: list[AgentMessage] = [
                        cast(AgentMessage, message) for message in injected_messages
                    ]
                    messages_to_send.append(
                        cast(
                            AgentMessage,
                            {"role": "user", "content": expanded, "timestamp": now_ms()},
                        )
                    )
                    preflight_ok = True
                    if preflight_result is not None:
                        preflight_result(True)
                    await self._agent.prompt(messages_to_send, current_images)
                finally:
                    self._agent.state.system_prompt = original_system_prompt
            else:
                preflight_ok = True
                if preflight_result is not None:
                    preflight_result(True)
                await self._agent.prompt(expanded, current_images)
            await self._check_compaction()
            await self._retry_failed_turn()
        except BaseException as exc:
            if not preflight_ok and preflight_result is not None:
                preflight_result(False)
            outcome = "aborted" if self._abort is not None and self._abort.is_set() else "failed"
            error = {"code": "error", "message": str(exc)}
            raise
        finally:
            await _finish_recorded_operation(self._session_manager, run_id, outcome, error)
            self._abort = None
            self._turn_timings.append(
                {
                    "startedAtMs": started_at_ms,
                    "durationMs": (time.perf_counter() - started) * 1000,
                }
            )

    async def _try_execute_extension_command(self, text: str) -> bool:
        """执行扩展注册的 /command；找到并执行返回 True（含 handler 报错）。"""
        runner = self._extension_runner
        if runner is None:
            return False
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        args = "" if space_index == -1 else text[space_index + 1 :]
        command = runner.get_registered_command(command_name)
        if command is None or command.handler is None:
            return False
        context = runner.create_command_context()
        try:
            result = command.handler(context, args)
            if inspect.isawaitable(result):
                await result
            return True
        except Exception as exc:
            from .extensions.types import ExtensionError

            runner.emit_error(
                ExtensionError(
                    extension_path=f"command:{command_name}",
                    event="command",
                    error=str(exc),
                )
            )
            return True

    def expand_prompt(self, text: str) -> str:
        """展开 `/skill:name` 与 `/templateName`；未匹配时原样返回。"""
        if not text.startswith("/"):
            return text
        expanded = self._expand_skill_command(text)
        if expanded != text:
            return expanded
        if self._template_loader is not None:
            expanded = self._template_loader.expand(text)
            if expanded != text:
                return expanded
        # 扩展动态提供的提示模板（resources_discover）。
        if self._extension_runner is not None:
            for template in self._extension_runner.get_discovered_prompts():
                prefix = f"/{template.name}"
                if text == prefix:
                    if self._template_loader is not None:
                        return self._template_loader.expand_template(template, "")
                    return template.content
                if text.startswith(prefix + " "):
                    args_string = text[len(prefix) :].strip()
                    if self._template_loader is not None:
                        return self._template_loader.expand_template(template, args_string)
                    return substitute_args(template.content, parse_command_args(args_string))
        return text

    def _expand_skill_command(self, text: str) -> str:
        """`/skill:name [instructions]` → <skill> XML 块（对齐 TS）。"""
        if not text.startswith("/skill:"):
            return text
        space_index = text.find(" ")
        skill_name = text[7:] if space_index == -1 else text[7:space_index]
        args = "" if space_index == -1 else text[space_index + 1 :].strip()
        if self._skill_loader is None:
            skill = None
        else:
            skill = self._skill_loader.get(skill_name)
        if skill is None and self._extension_runner is not None:
            skill = next(
                (
                    candidate
                    for candidate in self._extension_runner.get_discovered_skills()
                    if candidate.name == skill_name
                ),
                None,
            )
        if skill is None:
            return text
        try:
            body = strip_frontmatter(Path(skill.file_path).read_text(encoding="utf-8")).strip()
        except Exception as exc:
            if self._extension_runner is not None:
                # 延迟导入避免扩展类型与 Session 循环依赖。
                from .extensions.types import ExtensionError

                self._extension_runner.emit_error(
                    ExtensionError(
                        extension_path=skill.file_path,
                        event="skill_expansion",
                        error=str(exc),
                        stack=None,
                    )
                )
            return text
        block = (
            f'<skill name="{skill.name}" location="{skill.file_path}">\n'
            f"References are relative to {skill.base_dir}.\n\n{body}\n</skill>"
        )
        return f"{block}\n\n{args}" if args else block

    async def _retry_failed_turn(self) -> None:
        """turn 级自动重试：移除错误消息 + continue_() 恢复状态机。

        仅当本轮以可重试错误结束时触发；不重发原始请求。

        持久化说明：被移除的错误消息已在 message_end 时写入 JSONL
        （与 TS 一致：错误保留在 session 历史，仅从 agent state 弹出），
        重启后随 build_context() 恢复。
        """
        policy = self._turn_retry_policy
        if policy is None or not policy.enabled or policy.max_retries <= 0:
            return

        for attempt in range(1, policy.max_retries + 1):
            last = self._last_assistant_message()
            if last is None or last.get("stop_reason") != "error":
                return
            error_message = last.get("error_message")
            if not is_retryable_error(cast(str | None, error_message)):
                return

            # 移除错误消息，使末条消息非 assistant，满足 continue_() 约束
            self._agent.state._messages.pop()

            delay_ms = compute_backoff_delay(
                attempt,
                policy.base_delay_ms,
                max_delay_ms=policy.max_delay_ms,
                jitter=policy.jitter,
            )
            aborted = await self._abortable_sleep(delay_ms / 1000.0)
            if aborted:
                return
            await self._agent.continue_()

    def _last_assistant_message(self) -> AgentMessage | None:
        """返回 state 中最后一条消息（仅当为 assistant 时）。"""
        messages = self._agent.state.messages
        if not messages:
            return None
        last = messages[-1]
        if last.get("role") == "assistant":
            return last
        return None

    # ------------------------------------------------------------------
    # 自动压缩（对齐 TS _checkCompaction）
    # ------------------------------------------------------------------

    async def _check_compaction(self, skip_aborted_check: bool = True) -> bool:
        """检查是否需要压缩并在需要时执行。

        Case 1（溢出）：is_context_overflow 命中。
            - 未完成响应（error）→ 移除错误消息 + 压缩 + 自动重试
            - 已完成响应（stop）→ 压缩（不重试）
        Case 2（阈值）：should_compact 命中 → 压缩（不重试）。
        """
        settings = self._compaction_settings
        if not settings.enabled:
            return False

        assistant_message = self._last_assistant_message()
        if assistant_message is None:
            return False
        if skip_aborted_check and assistant_message.get("stop_reason") == "aborted":
            return False

        context_window = self._model.context_window if self._model else 0

        # 溢出检查仅在消息来自当前模型时进行：
        # 切换模型后旧模型的溢出错误不应触发新模型的压缩。
        same_model = (
            self._model is not None
            and assistant_message.get("provider") == self._model.provider
            and assistant_message.get("model") == self._model.id
        )

        # Case 1: 溢出。
        if same_model and is_context_overflow(
            cast(AssistantMessage, assistant_message), context_window
        ):
            will_retry = assistant_message.get("stop_reason") != "stop"

            if not will_retry:
                return await self._run_auto_compaction("overflow", False)

            if self._overflow_recovery_attempted:
                self._emit(
                    {
                        "type": "compaction_end",
                        "reason": "overflow",
                        "result": None,
                        "aborted": False,
                        "willRetry": False,
                        "errorMessage": (
                            "Context overflow recovery failed after one compact-and-retry "
                            "attempt. Try reducing context or switching to a larger-context model."
                        ),
                    }
                )
                return False

            self._overflow_recovery_attempted = True
            # 移除错误消息（仍保留在 session 历史，仅从 agent state 弹出）。
            messages = self._agent.state._messages
            if messages and messages[-1].get("role") == "assistant":
                self._agent.state._messages = messages[:-1]
            return await self._run_auto_compaction("overflow", will_retry)

        # Case 2: 阈值。
        # 对错误消息或全零 usage 消息，从最后一条有效响应估算上下文。
        usage = assistant_message.get("usage")
        direct_context_tokens = (
            calculate_context_tokens(cast(Usage, usage)) if isinstance(usage, dict) else 0
        )
        if assistant_message.get("stop_reason") == "error" or direct_context_tokens == 0:
            estimate = estimate_context_tokens(self._agent.state.messages)
            if estimate.last_usage_index is None:
                return False  # 完全没有 usage 数据
            context_tokens = estimate.tokens
        else:
            context_tokens = direct_context_tokens
        if should_compact(context_tokens, context_window, settings):
            return await self._run_auto_compaction("threshold", False)
        return False

    async def _run_auto_compaction(self, reason: str, will_retry: bool) -> bool:
        """执行自动压缩：prepare → 摘要 LLM → 追加压缩条目 → 重建上下文。

        will_retry=True（溢出恢复）时压缩后 continue_() 自动重试。
        """
        settings = self._compaction_settings
        if self._model is None:
            return False

        entries = self._session_manager.get_entries()
        context_messages = self._agent.state.messages
        preparation: CompactionPreparation | None = prepare_compaction(
            entries, context_messages, settings
        )
        if preparation is None:
            return False

        self._emit({"type": "compaction_start", "reason": reason})
        self._is_compacting = True
        try:
            cancelled, ext_result, from_extension = await self._run_compaction_hooks(
                preparation, reason, will_retry, None
            )
            if cancelled:
                self._emit(
                    {
                        "type": "compaction_end",
                        "reason": reason,
                        "result": None,
                        "aborted": True,
                        "willRetry": False,
                    }
                )
                return False
            if ext_result is None:
                summarization_auth = await self._get_summarization_request_auth(self._model)
                result = await compact(
                    preparation,
                    summarization_auth["model"],
                    api_key=summarization_auth.get("apiKey"),
                    headers=summarization_auth.get("headers"),
                    env=summarization_auth.get("env"),
                    stream_fn=self._agent._resolve_stream_fn(),
                    thinking_level=self._agent.state.thinking_level,
                )
            else:
                result = ext_result
        except Exception as exc:
            self._emit(
                {
                    "type": "compaction_end",
                    "reason": reason,
                    "result": None,
                    "aborted": False,
                    "willRetry": False,
                    "error": str(exc),
                }
            )
            return False
        finally:
            self._is_compacting = False

        entry_id = await self._session_manager.append_compaction(
            result.summary,
            result.first_kept_entry_id,
            result.tokens_before,
            result.details,
        )
        # 重建 agent 上下文（compactionSummary + 保留的近期消息）。
        self._agent.state.messages = self._session_manager.build_context()
        if will_retry:
            # v4 保留 tail 可能以 assistant 结尾；continue_ 要求末条非 assistant。
            messages = self._agent.state.messages
            while messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            self._agent.state.messages = messages
        self._emit_extension_event(
            "session_compact",
            {
                "type": "session_compact",
                "compactionEntry": {
                    "id": entry_id,
                    "type": "compaction",
                    "summary": result.summary,
                    "firstKeptEntryId": result.first_kept_entry_id,
                    "tokensBefore": result.tokens_before,
                },
                "fromExtension": from_extension,
                "reason": reason,
                "willRetry": will_retry,
            },
        )

        self._emit(
            {
                "type": "compaction_end",
                "reason": reason,
                "result": result,
                "aborted": False,
                "willRetry": will_retry,
            }
        )

        if will_retry:
            await self._agent.continue_()
            # 重试结果也需检查（对齐 TS：每次 agent_end 都检查压缩）。
            # _overflow_recovery_attempted 防止无限溢出恢复循环。
            await self._check_compaction()
        return True

    def _emit(self, event: dict[Any, Any]) -> None:
        """把会话级事件转发给所有外部监听器。"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # 监听器异常不应影响主流程

    async def _abortable_sleep(self, delay_s: float) -> bool:
        """可中止的退避等待（turn 级重试专用）。

        返回 True 表示被中止（应停止重试）。
        """
        if self._abort is None:
            await asyncio.sleep(delay_s)
            return False
        try:
            await asyncio.wait_for(self._abort.wait(), timeout=delay_s)
        except asyncio.TimeoutError:
            return False
        return True

    async def abort(self) -> None:
        """中止当前运行（含 turn 级重试的退避等待）。"""
        if self._abort is not None:
            self._abort.set()
        self._agent.abort()

    async def continue_(self) -> None:
        """从当前 transcript 继续（/input 编辑消息并重建会话后重跑分支用）。"""
        run_id = await _record_operation(self._session_manager, "run")
        try:
            await self._agent.continue_()
        except BaseException as exc:
            await _finish_recorded_operation(
                self._session_manager,
                run_id,
                "failed",
                {"code": "error", "message": str(exc)},
            )
            raise
        await _finish_recorded_operation(self._session_manager, run_id)

    async def recovery_state(self) -> str:
        """当前会话恢复状态：idle / suspended / corrupt（无此能力时恒为 idle）。"""
        check = getattr(self._session_manager, "recovery_state", None)
        if check is None:
            return "idle"
        return await check()

    async def open_operations(self, limit: int = 2) -> list[dict]:
        """当前未完成的操作记录（无此能力时返回空）。"""
        operations = getattr(self._session_manager, "open_operations", None)
        if operations is None:
            return []
        return await operations(limit=limit)

    async def resume_suspended_operation(self) -> bool:
        """检测挂起的 run 并重放原始 prompt；无挂起或无法重放返回 False。"""
        if await self.recovery_state() != "suspended":
            return False
        operations = await self.open_operations(limit=1)
        if not operations:
            return False
        operation = operations[0]
        intent = operation.get("intent") or {}
        if intent.get("kind") != "run":
            return False
        text: str | None = None
        for message in intent.get("originalPrompt") or []:
            content = message.get("content")
            if isinstance(content, str):
                text = content
                break
        if text is None:
            return False
        finish = getattr(self._session_manager, "finish_operation", None)
        if finish is not None:
            try:
                await finish(
                    operation["id"],
                    outcome="aborted",
                    error={"code": "crash", "message": "resumed after suspend"},
                )
            except Exception:
                pass
        await self.prompt(text)
        return True

    async def fetch_deferred(self) -> bool:
        """抓取最后一条 deferred assistant 消息的结果并追加到会话。

        返回 True 表示已抓取；随后可调用 continue_() 继续。
        """
        message = self._last_deferred_message()
        handle = message.get("deferred") if message is not None else None
        if (
            not handle
            or self._model is None
            or self._model_runtime is None
            or not isinstance(handle, dict)
        ):
            return False
        result = await self._model_runtime.fetch_deferred(self._model, cast(DeferredHandle, handle))
        self._agent.state._append_message(result)
        self._persist_message(result)
        record = getattr(self._session_manager, "record_write_deferred", None)
        if record is not None:
            try:
                await record(
                    {
                        "type": "message",
                        "id": handle.get("id", ""),
                        "message": message,
                    }
                )
            except Exception:
                pass
        return True

    async def cancel_deferred(self) -> bool:
        """取消最后一条 deferred assistant 消息（若支持）。"""
        message = self._last_deferred_message()
        handle = message.get("deferred") if message is not None else None
        if (
            not handle
            or self._model is None
            or self._model_runtime is None
            or not isinstance(handle, dict)
        ):
            return False
        await self._model_runtime.cancel_deferred(self._model, cast(DeferredHandle, handle))
        return True

    def _last_deferred_message(self) -> AgentMessage | None:
        """返回最近一条带 deferred 句柄的 assistant 消息。"""
        for message in reversed(self._agent.state.messages):
            if message.get("role") == "assistant" and isinstance(message.get("deferred"), dict):
                return message
        return None

    @property
    def is_bash_running(self) -> bool:
        """是否有交互 bash 命令正在运行（一次仅允许一条）。"""
        return bool(self._bash_abort_signals)

    def abort_bash(self) -> None:
        """中止正在运行的交互 bash 命令。"""
        for signal in list(self._bash_abort_signals):
            signal.set()

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str, Any], None] | None = None,
        *,
        exclude_from_context: bool = False,
        shell_path: str | None = None,
        command_prefix: str | None = None,
        timeout: float | None = None,
    ) -> BashResult:
        """执行交互 shell 命令（对齐 TS AgentSession.executeBash）。

        `!cmd` 的结果会作为 bashExecution 消息进入 LLM 上下文；
        `!!cmd`（exclude_from_context=True）结果仅展示，不进入上下文。
        运行期间可用 abort_bash() 中止；同时只允许一条 bash。
        """
        runner = self._extension_runner
        if runner is not None and runner.has_handlers("user_bash"):
            results = await runner.emit_event(
                "user_bash",
                {
                    "type": "user_bash",
                    "command": command,
                    "excludeFromContext": exclude_from_context,
                    "cwd": self._cwd,
                },
            )
            for result in reversed(results):
                if not isinstance(result, dict):
                    continue
                if isinstance(result.get("result"), dict):
                    data = result["result"]
                    self.record_bash_result(
                        command, data, exclude_from_context=exclude_from_context
                    )
                    return BashResult(
                        output=data.get("output", ""),
                        exit_code=data.get("exitCode", 0),
                        cancelled=bool(data.get("cancelled", False)),
                        truncated=bool(data.get("truncated", False)),
                        full_output_path=data.get("fullOutputPath"),
                    )
                operations = result.get("operations")
                if isinstance(operations, dict) and callable(operations.get("exec")):
                    op_result = operations["exec"](command, self._cwd, {"timeout": timeout})
                    if inspect.isawaitable(op_result):
                        op_result = await op_result
                    if not isinstance(op_result, dict):
                        break
                    self.record_bash_result(
                        command, op_result, exclude_from_context=exclude_from_context
                    )
                    return BashResult(
                        output=op_result.get("output", ""),
                        exit_code=op_result.get("exitCode", 0),
                        cancelled=bool(op_result.get("cancelled", False)),
                        truncated=bool(op_result.get("truncated", False)),
                        full_output_path=op_result.get("fullOutputPath"),
                    )
        if self.is_bash_running:
            raise RuntimeError("A bash command is already running")
        abort_signal = asyncio.Event()
        self._bash_abort_signals.add(abort_signal)
        env = PythonExecutionEnv(self._cwd, shell_path=shell_path)
        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command
        try:
            ok, result = await execute_shell_with_capture(
                env,
                resolved_command,
                {
                    "cwd": self._cwd,
                    "inheritEnv": True,
                    "timeout": timeout,
                    "abortSignal": abort_signal,
                    "onChunk": on_chunk,
                    "returnExecutionErrors": True,
                },
            )
            if not ok:
                raise result
            if result.execution_error is not None:
                raise result.execution_error
            self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            return BashResult(
                output=result.output,
                exit_code=result.exit_code,
                cancelled=result.cancelled,
                truncated=result.truncation.truncated,
                full_output_path=result.full_output_path,
            )
        finally:
            self._bash_abort_signals.discard(abort_signal)

    def record_bash_result(
        self,
        command: str,
        result: Any,
        *,
        exclude_from_context: bool = False,
    ) -> None:
        """把 bashExecution 消息写入会话历史/上下文（对齐 TS recordBashResult）。"""
        if isinstance(result, dict):
            output = result.get("output", "")
            exit_code = result.get("exitCode", 0)
            cancelled = bool(result.get("cancelled", False))
            truncated = bool(result.get("truncated", False))
            full_output_path = result.get("fullOutputPath")
        else:
            output = result.output
            exit_code = result.exit_code
            cancelled = result.cancelled
            truncated = result.truncation.truncated
            full_output_path = result.full_output_path
        message: dict[str, Any] = {
            "role": "bashExecution",
            "command": command,
            "output": output,
            "exitCode": exit_code,
            "cancelled": cancelled,
            "truncated": truncated,
            "fullOutputPath": full_output_path,
            "timestamp": now_ms(),
            "excludeFromContext": exclude_from_context,
        }
        if self.is_streaming:
            # Agent 运行中延迟写入，避免打断 tool_use/tool_result 顺序。
            self._pending_bash_messages.append(message)
            return
        self._agent.state._append_message(cast(AgentMessage, message))
        self._persist_message(cast(AgentMessage, message))

    def _flush_pending_bash_messages(self) -> None:
        """Agent 一轮结束后把延迟的 bashExecution 消息写入上下文。"""
        if not self._pending_bash_messages:
            return
        pending = self._pending_bash_messages
        self._pending_bash_messages = []
        for message in pending:
            bash_message = cast(AgentMessage, message)
            self._agent.state._append_message(bash_message)
            self._persist_message(bash_message)

    def _persist_message(self, message: AgentMessage) -> None:
        """后台把消息写入 JSONL，dispose 时等待写入完成。"""
        task = asyncio.create_task(self._session_manager.append_message(message))
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)
        if message.get("role") == "assistant" and isinstance(message.get("usage"), dict):

            async def _record_usage() -> None:
                try:
                    entry_id = await task
                except Exception:
                    return
                await _record_usage_for_message(self._session_manager, message, entry_id)

            usage_task = asyncio.create_task(_record_usage())
            self._pending_writes.add(usage_task)
            usage_task.add_done_callback(self._pending_writes.discard)

    async def wait_for_idle(self) -> None:
        """等待当前运行结束（含所有事件监听器完成）。"""
        await self._agent.wait_for_idle()

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """订阅 Agent 生命周期事件。返回取消订阅函数。"""
        self._listeners.append(cast(Callable[[dict[Any, Any]], None], listener))

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(cast(Callable[[dict[Any, Any]], None], listener))
            except ValueError:
                pass

        return _unsubscribe

    def get_messages(self) -> list[AgentMessage]:
        """获取当前会话的所有消息。"""
        return self._agent.state.messages

    async def dispose(self) -> None:
        """销毁会话：等待 pending writes → 取消订阅 + 中止运行。"""
        if self._extension_runner is not None:
            try:
                await self._extension_runner.shutdown_all()
            except Exception:
                pass
        # 等待所有后台持久化写入完成
        if self._pending_writes:
            await asyncio.gather(*self._pending_writes, return_exceptions=True)
            self._pending_writes.clear()
        self.abort_bash()
        self._agent.abort()
        try:
            await self._agent.wait_for_idle()
        except Exception:
            pass
        if self._unsub_agent:
            self._unsub_agent()
            self._unsub_agent = None
        self._listeners.clear()
        # 统一清理该会话注册的资源（不阻断 dispose 自身）。
        from pi_ai.session_resources import cleanup_session_resources

        try:
            cleanup_session_resources(self._session_manager.session_id)
        except Exception:
            pass
        close = getattr(self._session_manager, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 内部：事件桥接
    # ------------------------------------------------------------------

    async def _handle_agent_event(
        self,
        event: AgentEvent,
        signal: asyncio.Event | None = None,
    ) -> None:
        """Agent 事件 → 持久化到 SessionManager + 转发给外部监听器。

        在 message_end 时自动写入 JSONL。
        signal 为 Agent 当前运行的 abort signal（1.3 监听器协议），当前忽略。
        """
        # 扩展事件转发（Agent 生命周期 / 消息 / 工具钩子）。
        if self._extension_runner is not None:
            try:
                await self._extension_runner.emit_event(event.get("type", ""), event)
            except Exception:
                pass

        event_type = event.get("type")

        # message_end → 持久化
        if event_type == "message_end":
            msg = event.get("message")
            if msg is not None:
                self._persist_message(cast(AgentMessage, msg))

        # agent_end / agent_settled → 刷入延迟的 bashExecution 消息。
        if event_type in ("agent_end", "agent_settled"):
            self._flush_pending_bash_messages()

        # 转发给所有外部监听器
        for listener in self._listeners:
            try:
                listener(cast(dict[Any, Any], event))
            except Exception:
                pass  # 监听器异常不应影响主流程
