"""ExtensionRunner（5.3 + 5.4）——生命周期、事件分发、ExtensionContext。"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Callable

from .types import (
    EventBus,
    Extension,
    ExtensionError,
    ExtensionFlag,
    ExtensionRuntime,
    ExtensionShortcut,
    NoopUIContext,
    RegisteredCommand,
    ToolDefinition,
)


class _ModelRegistryAdapter:
    """ctx.modelRegistry：find(provider, id) / complete(model, context, options)。"""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def find(self, provider: str, model_id: str):
        if self._runtime is None:
            return None
        return self._runtime.get_model(provider, model_id)

    async def complete(self, model, context, options=None):
        if self._runtime is None:
            raise RuntimeError("No model runtime available")
        return await self._runtime.complete(model, context, options)


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------


class ExtensionContext:
    """事件处理器收到的上下文（值在访问时实时解析）。

    stale-runtime 保护（对齐 TS）：创建时记录 runtime generation，
    会话替换 / reload（runtime.invalidate）后继续使用被捕获的旧 ctx 会抛错。
    """

    def __init__(self, runner: "ExtensionRunner") -> None:
        self._runner = runner
        self._generation = runner.runtime._generation

    def _assert_active(self) -> None:
        self._runner.runtime.assert_active(self._generation)

    @property
    def ui(self):
        self._assert_active()
        return self._runner.ui_context

    @property
    def mode(self) -> str:
        self._assert_active()
        return self._runner.mode

    @property
    def cwd(self) -> str:
        self._assert_active()
        return self._runner.cwd

    @property
    def session(self):
        self._assert_active()
        return self._runner.session

    @property
    def session_manager(self):
        """会话树管理器（对齐 TS ctx.sessionManager）。"""
        self._assert_active()
        session = self._runner.session
        return session.session_manager if session is not None else None

    @property
    def signal(self):
        """当前 turn 的中止信号（无运行 turn 时为 None）。"""
        self._assert_active()
        session = self._runner.session
        return getattr(session, "_abort", None) if session is not None else None

    @property
    def model(self):
        self._assert_active()
        return self._runner.session.model if self._runner.session is not None else None

    @property
    def thinking_level(self):
        self._assert_active()
        if self._runner.session is None:
            return None
        return self._runner.session.thinking_level

    @property
    def scoped_models(self):
        """--models 循环列表（对齐 TS ctx.scopedModels）。"""
        self._assert_active()
        session = self._runner.session
        return list(session.scoped_models) if session is not None else []

    def is_project_trusted(self) -> bool:
        """当前项目是否被信任（未解析时返回 False）。"""
        self._assert_active()
        session = self._runner.session
        return bool(session.project_trusted) if session is not None else False

    def is_idle(self) -> bool:
        self._assert_active()
        return not (self._runner.session is not None and self._runner.session.is_streaming)

    @property
    def has_ui(self) -> bool:
        """是否运行在有 UI 的上下文（TUI / RPC）；print 模式为 False。"""
        self._assert_active()
        return self._runner.mode != "print"

    @property
    def model_registry(self) -> _ModelRegistryAdapter:
        """模型注册表（find / complete），供扩展自选摘要模型等使用。"""
        self._assert_active()
        return _ModelRegistryAdapter(self._runner.model_runtime)

    def has_pending_messages(self) -> bool:
        self._assert_active()
        return bool(self._runner.session is not None and self._runner.session.pending_message_count)

    def abort(self) -> None:
        self._assert_active()
        self._runner.abort()

    def shutdown(self) -> None:
        self._assert_active()
        self._runner.shutdown()

    async def compact(self, options: dict | None = None) -> None:
        """触发压缩（对齐 TS compact(options)：不等待完成，onComplete/onError 回调）。"""
        self._assert_active()
        options = options or {}
        custom_instructions = options.get("customInstructions")
        on_complete = options.get("onComplete")
        on_error = options.get("onError")

        async def _run() -> None:
            try:
                result = None
                if self._runner.session is not None:
                    result = await self._runner.session.compact(custom_instructions)
                if callable(on_complete):
                    on_complete(result)
            except Exception as exc:
                if callable(on_error):
                    on_error(exc)

        self._runner._schedule(_run())

    def get_system_prompt(self) -> str:
        self._assert_active()
        session = self._runner.session
        if session is None:
            return ""
        return session._agent.state.system_prompt

    def get_system_prompt_options(self) -> dict:
        """当前系统提示构建相关选项（对齐 TS getSystemPromptOptions 的最小集）。"""
        self._assert_active()
        session = self._runner.session
        if session is None:
            return {}
        return {
            "systemPrompt": session._agent.state.system_prompt,
            "cwd": self._runner.cwd,
            "model": session.model,
            "thinkingLevel": session.thinking_level,
        }

    def get_context_usage(self):
        """估算当前上下文 token 用量（对齐 TS getContextUsage）。

        以最后一条带 usage 的 assistant 消息为基准，加上其后的消息估算；
        完全没有 usage 数据时返回 None。
        """
        self._assert_active()
        session = self._runner.session
        if session is None:
            return None
        from pi_agent.compaction_utils import estimate_context_tokens

        estimate = estimate_context_tokens(session._agent.state.messages)
        if estimate.last_usage_index is None:
            return None
        return {"tokens": estimate.tokens}


class ExtensionCommandContext(ExtensionContext):
    """命令处理器上下文：会话控制方法。"""

    async def wait_for_idle(self) -> None:
        if self._runner.session is not None:
            await self._runner.session.wait_for_idle()

    async def _cancel_requested(self, event_type: str, data: dict) -> bool:
        """派发会话替换前置事件；任一 handler 返回 cancel 则阻止。"""
        runner = self._runner
        if runner is None or not runner.has_handlers(event_type):
            return False
        results = await runner.emit_event(event_type, {"type": event_type, **data})
        for result in reversed(results):
            if isinstance(result, dict) and result.get("cancel"):
                return True
        return False

    async def new_session(self, options: dict | None = None):
        options = options or {}
        if await self._cancel_requested(
            "session_before_switch",
            {"position": "before", "targetSessionFile": None},
        ):
            return None
        result = await self._runner._command_action("new_session", options)
        await self._run_with_session(options)
        return result

    async def fork(self, entry_id: str, options: dict | None = None):
        options = options or {}
        if await self._cancel_requested(
            "session_before_fork",
            {"position": "before", "entryId": entry_id},
        ):
            return None
        result = await self._runner._command_action("fork", entry_id, options)
        await self._run_with_session(options)
        return result

    async def navigate_tree(self, target_id: str, options: dict | None = None):
        return await self._runner._command_action("navigate_tree", target_id, options or {})

    async def switch_session(self, session_path: str, options: dict | None = None):
        options = options or {}
        if await self._cancel_requested(
            "session_before_switch",
            {"position": "at", "targetSessionFile": session_path},
        ):
            return None
        result = await self._runner._command_action("switch_session", session_path, options)
        await self._run_with_session(options)
        return result

    async def reload(self) -> None:
        await self._runner._command_action("reload")

    async def _run_with_session(self, options: dict) -> None:
        """会话替换后执行 withSession 回调（对齐 TS ReplacedSessionContext）。

        回调收到绑定到新会话的全新 command ctx（runtime 已 invalidate，
        新 ctx 是当前 generation，可直接使用）。
        """
        with_session = options.get("withSession")
        if not callable(with_session):
            return
        fresh = self._runner.create_command_context()
        raw = with_session(fresh)
        if inspect.isawaitable(raw):
            await raw


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExtensionRunner:
    """扩展生命周期与事件分发。"""

    def __init__(
        self,
        extensions: list[Extension] | None = None,
        runtime: ExtensionRuntime | None = None,
        *,
        cwd: str = "",
        session=None,
        model_runtime=None,
    ) -> None:
        self.extensions = list(extensions or [])
        self.runtime = runtime or ExtensionRuntime()
        self.cwd = cwd
        self.session = session
        self.model_runtime = model_runtime
        self.ui_context = NoopUIContext()
        self.mode = "print"
        self.event_bus = EventBus()

        self._error_listeners: list[Callable[[ExtensionError], None]] = []
        self._background_tasks: set[asyncio.Task] = set()
        self._discovered_skills: list = []
        self._discovered_prompts: list = []
        self._discovered_themes: list = []
        self._shutdown_handler: Callable[[], None] | None = None
        self._abort_fn: Callable[[], None] | None = None
        self._command_handlers: dict[str, Callable] = {}

    def _schedule(self, coro) -> None:
        """调度后台任务并持有引用（防止被 GC 且便于测试等待）。"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------
    # 绑定
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        ui_context=None,
        mode: str | None = None,
        session=None,
        model_runtime=None,
        shutdown_handler: Callable[[], None] | None = None,
        abort_fn: Callable[[], None] | None = None,
        command_handlers: dict[str, Callable] | None = None,
        actions: dict[str, Callable] | None = None,
    ) -> None:
        if ui_context is not None:
            self.ui_context = ui_context
        if mode is not None:
            self.mode = mode
        if session is not None:
            self.session = session
        if model_runtime is not None:
            self.model_runtime = model_runtime
        if shutdown_handler is not None:
            self._shutdown_handler = shutdown_handler
        if abort_fn is not None:
            self._abort_fn = abort_fn
        if command_handlers:
            self._command_handlers.update(command_handlers)
        for name, fn in (actions or {}).items():
            self.runtime.set_action(name, fn)

    def bind_session(self, session) -> None:
        """会话就绪后绑定（AgentSession 构造时调用）。"""
        self.session = session
        self.runtime.set_action("set_model", self._action_set_model)
        self.runtime.set_action("get_thinking_level", lambda: session.thinking_level)
        self.runtime.set_action("set_thinking_level", session.set_thinking_level)
        self.runtime.set_action("set_session_name", session.set_session_name)
        self.runtime.set_action("get_session_name", lambda: session.session_name)
        self.runtime.set_action(
            "send_user_message",
            lambda content, options: self._action_send_user_message(session, content, options),
        )
        self.runtime.set_action("get_active_tools", lambda: self._get_active_tools(session))
        self.runtime.set_action(
            "set_active_tools", lambda names: self._set_active_tools(session, names)
        )
        self.runtime.set_action("get_all_tools", lambda: self._get_all_tools(session))
        self.runtime.set_action("get_commands", self.get_registered_commands)
        self.runtime.set_action(
            "send_message",
            lambda content, options: self._action_send_message(session, content, options),
        )
        self.runtime.set_action(
            "append_entry",
            lambda custom_type, data: self._action_append_entry(session, custom_type, data),
        )
        self.runtime.set_action(
            "set_label", lambda entry_id, label: self._action_set_label(session, entry_id, label)
        )
        self.apply_providers()

    # ------------------------------------------------------------------
    # 后台动作（send_message / append_entry / set_label）
    # ------------------------------------------------------------------

    def _action_send_message(self, session, content, options: dict | None = None) -> None:
        self._schedule(self._run_send_message(session, content, options or {}))

    async def _run_send_message(self, session, content, options: dict) -> None:
        custom_type = options.get("customType") or "extension"
        await session._session_manager.append_custom_message_entry(
            custom_type,
            content,
            display=bool(options.get("display", True)),
            details=options.get("details"),
        )
        deliver = options.get("deliverAs")
        if deliver not in ("followUp", "steer"):
            return
        if isinstance(content, dict) and isinstance(content.get("content"), str):
            text = content["content"]
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        await self._deliver_with_input(session, text, deliver)
        if deliver == "followUp" and options.get("triggerTurn"):
            try:
                if not session.is_streaming:
                    await session.continue_()
            except Exception:
                pass

    def _action_append_entry(self, session, custom_type: str, data) -> None:
        self._schedule(session._session_manager.append_custom_entry(custom_type, data))

    def _action_set_label(self, session, entry_id: str, label: str | None) -> None:
        session._session_manager.set_label(entry_id, label)

    # ------------------------------------------------------------------
    # 错误
    # ------------------------------------------------------------------

    def on_error(self, listener: Callable[[ExtensionError], None]) -> Callable[[], None]:
        self._error_listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._error_listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    def emit_error(self, error: ExtensionError) -> None:
        for listener in list(self._error_listeners):
            try:
                listener(error)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 事件分发
    # ------------------------------------------------------------------

    def has_handlers(self, event_type: str) -> bool:
        return any(extension.handlers.get(event_type) for extension in self.extensions)

    async def emit_event(self, event_type: str, data: dict | None = None) -> list[Any]:
        """按注册顺序分发事件，收集处理器返回值（单个失败不中断）。"""
        context = self.create_context()
        event = {"type": event_type, **(data or {})}
        results: list[Any] = []
        for extension in self.extensions:
            for handler in list(extension.handlers.get(event_type, [])):
                try:
                    result = handler(event, context)
                    if inspect.isawaitable(result):
                        result = await result
                    results.append(result)
                except Exception as exc:
                    self.emit_error(
                        ExtensionError(
                            extension_path=extension.path,
                            event=event_type,
                            error=str(exc),
                        )
                    )
        return results

    async def emit_input(
        self,
        text: str,
        *,
        images=None,
        source: str = "interactive",
        streaming_behavior: str | None = None,
    ) -> tuple[str, str | None]:
        """input 事件链：transform 更新文本；handled 短路。返回 (text, action)。"""
        context = self.create_context()
        current_text = text
        for extension in self.extensions:
            for handler in list(extension.handlers.get("input", [])):
                try:
                    event = {
                        "type": "input",
                        "text": current_text,
                        "images": images,
                        "source": source,
                        "streamingBehavior": streaming_behavior,
                    }
                    result = handler(event, context)
                    if inspect.isawaitable(result):
                        result = await result
                    if not isinstance(result, dict):
                        continue
                    action = result.get("action")
                    if action == "handled":
                        return result.get("text", current_text), "handled"
                    if action == "transform":
                        current_text = result.get("text", current_text)
                except Exception as exc:
                    self.emit_error(
                        ExtensionError(
                            extension_path=extension.path,
                            event="input",
                            error=str(exc),
                        )
                    )
        return current_text, "continue"

    async def emit_project_trust(self, cwd: str) -> str | None:
        """project_trust 事件：返回首个 yes/no 决定；undecided 继续。"""
        context = self.create_context()
        event = {"type": "project_trust", "cwd": cwd}
        for extension in self.extensions:
            for handler in list(extension.handlers.get("project_trust", [])):
                try:
                    result = handler(event, context)
                    if inspect.isawaitable(result):
                        result = await result
                    if isinstance(result, dict) and result.get("trusted") in (
                        "yes",
                        "no",
                        "undecided",
                    ):
                        if result["trusted"] != "undecided":
                            return result["trusted"]
                except Exception as exc:
                    self.emit_error(
                        ExtensionError(
                            extension_path=extension.path,
                            event="project_trust",
                            error=str(exc),
                        )
                    )
        return None

    # ------------------------------------------------------------------
    # 动态资源（resources_discover）
    # ------------------------------------------------------------------

    async def discover_resources(self) -> None:
        """resources_discover 事件：收集扩展动态提供的 skills / prompts / themes。"""
        self._discovered_skills = []
        self._discovered_prompts = []
        self._discovered_themes = []
        if not self.has_handlers("resources_discover"):
            return
        results = await self.emit_event("resources_discover", {"type": "resources_discover"})
        for result in results:
            if not isinstance(result, dict):
                continue
            for key, target in (
                ("skills", self._discovered_skills),
                ("prompts", self._discovered_prompts),
                ("themes", self._discovered_themes),
            ):
                items = result.get(key)
                if isinstance(items, list):
                    target.extend(items)
            # TS 兼容：路径形式（skillPaths / promptPaths / themePaths）。
            from ..prompt_templates import _load_template_from_file
            from ..skills import _load_skill_from_file

            for raw in result.get("skillPaths") or []:
                skill, _diagnostics = _load_skill_from_file(Path(raw), "extension")
                if skill is not None:
                    self._discovered_skills.append(skill)
            for raw in result.get("promptPaths") or []:
                template = _load_template_from_file(Path(raw), "extension")
                if template is not None:
                    self._discovered_prompts.append(template)
            for raw in result.get("themePaths") or []:
                from pi_tui.theme import ThemeLoader

                path = Path(raw)
                try:
                    theme = ThemeLoader(path.parent).load(path.stem)
                except Exception:
                    continue
                self._discovered_themes.append(theme)

    def get_discovered_skills(self) -> list:
        return list(self._discovered_skills)

    def get_discovered_prompts(self) -> list:
        return list(self._discovered_prompts)

    def get_discovered_themes(self) -> list:
        return list(self._discovered_themes)

    # ------------------------------------------------------------------
    # 上下文
    # ------------------------------------------------------------------

    def create_context(self) -> ExtensionContext:
        return ExtensionContext(self)

    def create_command_context(self) -> ExtensionCommandContext:
        return ExtensionCommandContext(self)

    async def _command_action(self, name: str, *args):
        handler = self._command_handlers.get(name)
        if handler is None:
            raise NotImplementedError(f"Command action '{name}' is not available in this mode")
        result = handler(*args)
        if inspect.isawaitable(result):
            result = await result
        # 会话替换 / reload 后使旧 ctx 过期（对齐 TS runner.invalidate）。
        if name in ("new_session", "fork", "switch_session", "reload"):
            self.runtime.invalidate()
        return result

    # ------------------------------------------------------------------
    # 注册项聚合
    # ------------------------------------------------------------------

    def get_registered_tools(self) -> list[ToolDefinition]:
        tools_by_name: dict[str, ToolDefinition] = {}
        for extension in self.extensions:
            for name, tool in extension.tools.items():
                tools_by_name.setdefault(name, tool)
        return list(tools_by_name.values())

    def get_tool_definition(self, tool_name: str) -> ToolDefinition | None:
        for extension in self.extensions:
            tool = extension.tools.get(tool_name)
            if tool is not None:
                return tool
        return None

    def get_registered_commands(self) -> list[RegisteredCommand]:
        """聚合命令；同名命令以 `name:1`、`name:2` 区分调用名。"""
        commands: list[RegisteredCommand] = []
        counts: dict[str, int] = {}
        for extension in self.extensions:
            for command in extension.commands.values():
                commands.append(command)
                counts[command.name] = counts.get(command.name, 0) + 1
        seen: dict[str, int] = {}
        result: list[RegisteredCommand] = []
        for command in commands:
            occurrence = seen.get(command.name, 0) + 1
            seen[command.name] = occurrence
            invocation = (
                command.name if counts[command.name] == 1 else f"{command.name}:{occurrence}"
            )
            result.append(
                RegisteredCommand(
                    name=invocation,
                    description=command.description,
                    argument_hint=command.argument_hint,
                    get_argument_completions=getattr(
                        command,
                        "get_argument_completions",
                        None,
                    ),
                    handler=command.handler,
                    source_info=command.source_info,
                )
            )
        return result

    def get_flags(self) -> list[ExtensionFlag]:
        flags_by_name: dict[str, ExtensionFlag] = {}
        for extension in self.extensions:
            for name, flag in extension.flags.items():
                flags_by_name.setdefault(name, flag)
        return list(flags_by_name.values())

    def set_flag_value(self, name: str, value: bool | str | None) -> None:
        """写入扩展 flag 的运行时值（CLI 两段解析后调用）。"""
        self.runtime.flag_values[name] = value

    def get_shortcuts(self) -> list[ExtensionShortcut]:
        shortcuts: list[ExtensionShortcut] = []
        for extension in self.extensions:
            shortcuts.extend(extension.shortcuts.values())
        return shortcuts

    def get_message_renderer(self, custom_type: str):
        for extension in self.extensions:
            renderer = extension.message_renderers.get(custom_type)
            if renderer is not None:
                return renderer
        return None

    def get_tool_renderer(self, tool_name: str):
        for extension in self.extensions:
            renderer = extension.tool_renderers.get(tool_name)
            if renderer is not None:
                return renderer
        return None

    def get_entry_renderer(self, custom_type: str):
        for extension in self.extensions:
            renderer = extension.entry_renderers.get(custom_type)
            if renderer is not None:
                return renderer
        return None

    def get_markdown_transformers(self) -> list[Callable]:
        transformers: list[Callable] = []
        for extension in self.extensions:
            transformers.extend(extension.markdown_transformers)
        return transformers

    def get_autocomplete(self) -> list[Callable]:
        providers: list[Callable] = []
        for extension in self.extensions:
            providers.extend(extension.autocomplete)
        return providers

    # ------------------------------------------------------------------
    # Provider / 工具应用
    # ------------------------------------------------------------------

    def apply_providers(self) -> None:
        """把扩展注册的 provider 配置应用到 ModelRuntime。"""
        if self.model_runtime is None:
            return
        for extension in self.extensions:
            for name, config in extension.providers:
                try:
                    self.model_runtime.register_provider(name, config)
                except Exception as exc:
                    self.emit_error(
                        ExtensionError(
                            extension_path=extension.path,
                            event="register_provider",
                            error=str(exc),
                        )
                    )

    def _get_active_tools(self, session) -> list[str]:
        if session is None:
            return []
        return [tool.name for tool in session._agent.state.tools]

    def _set_active_tools(self, session, tool_names: list[str]) -> None:
        if session is None:
            return
        current = session._agent.state.tools
        by_name = {tool.name: tool for tool in current}
        tools = [by_name[name] for name in tool_names if name in by_name]
        session._agent.state.tools = tools
        if session.extension_state is not None:
            session.extension_state["active_tools"] = list(tools)
        session.rebuild_system_prompt()

    def _get_all_tools(self, session) -> list[dict]:
        if session is None:
            return []
        result: list[dict] = []
        for tool in session._agent.state.tools:
            definition = self.get_tool_definition(tool.name)
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "prompt_guidelines": tool.prompt_guidelines,
                    "source_info": (definition.source_info if definition is not None else None)
                    or {},
                }
            )
        return result

    async def _action_set_model(self, model) -> bool:
        if self.session is None:
            return False
        try:
            await self.session.set_model(model)
            return True
        except Exception:
            return False

    def _action_send_user_message(self, session, content, options) -> None:
        if session is None:
            return
        text = content if isinstance(content, str) else ""
        if options.get("deliverAs") == "followUp":
            self._schedule(self._deliver_with_input(session, text, "followUp"))
        elif options.get("deliverAs") == "steer":
            self._schedule(self._deliver_with_input(session, text, "steer"))
        else:
            self._schedule(self._send_prompt_text(session, text))

    async def _deliver_with_input(self, session, text: str, behavior: str) -> None:
        """按 deliverAs 投递，并把 input 事件的 streamingBehavior 透传给扩展。"""
        if session._extension_runner is not None and session._extension_runner.has_handlers(
            "input"
        ):
            text, _action = await session._extension_runner.emit_input(
                text,
                source="extension",
                streaming_behavior=behavior,
            )
        if behavior == "followUp":
            session.follow_up(text)
        else:
            session.steer(text)

    async def _send_prompt_text(self, session, text: str) -> None:
        await session.prompt(text)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def abort(self) -> None:
        if self._abort_fn is not None:
            self._abort_fn()
            return
        if self.session is not None:
            try:
                asyncio.get_running_loop().create_task(self.session.abort())
            except RuntimeError:
                pass

    def shutdown(self) -> None:
        if self._shutdown_handler is not None:
            self._shutdown_handler()

    async def shutdown_all(self) -> None:
        """卸载：分发 session_shutdown，等待并清理后台任务。"""
        if self.has_handlers("session_shutdown"):
            await self.emit_event("session_shutdown")
        # 等待已调度的后台任务完成（shutdown 分发期间新调度的也会被取消）。
        tasks = list(self._background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        remaining = list(self._background_tasks)
        for task in remaining:
            task.cancel()
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


__all__ = [
    "ExtensionRunner",
    "ExtensionContext",
    "ExtensionCommandContext",
]
