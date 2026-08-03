"""ExtensionRunner（5.3 + 5.4）——生命周期、事件分发、ExtensionContext。"""

from __future__ import annotations

import inspect
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


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------


class ExtensionContext:
    """事件处理器收到的上下文（值在访问时实时解析）。"""

    def __init__(self, runner: "ExtensionRunner") -> None:
        self._runner = runner

    @property
    def ui(self):
        return self._runner.ui_context

    @property
    def mode(self) -> str:
        return self._runner.mode

    @property
    def cwd(self) -> str:
        return self._runner.cwd

    @property
    def session(self):
        return self._runner.session

    @property
    def model(self):
        return self._runner.session.model if self._runner.session is not None else None

    @property
    def thinking_level(self):
        if self._runner.session is None:
            return None
        return self._runner.session.thinking_level

    def is_idle(self) -> bool:
        return not (
            self._runner.session is not None and self._runner.session.is_streaming
        )

    def has_pending_messages(self) -> bool:
        return bool(
            self._runner.session is not None and self._runner.session.pending_message_count
        )

    def abort(self) -> None:
        self._runner.abort()

    def shutdown(self) -> None:
        self._runner.shutdown()

    async def compact(self) -> None:
        if self._runner.session is not None:
            await self._runner.session.compact()

    def get_system_prompt(self) -> str:
        session = self._runner.session
        if session is None:
            return ""
        return session._agent.state.system_prompt

    def get_context_usage(self):
        return None


class ExtensionCommandContext(ExtensionContext):
    """命令处理器上下文：会话控制方法。"""

    async def wait_for_idle(self) -> None:
        if self._runner.session is not None:
            await self._runner.session.wait_for_idle()

    async def new_session(self, options: dict | None = None):
        return await self._runner._command_action("new_session", options or {})

    async def fork(self, entry_id: str, options: dict | None = None):
        return await self._runner._command_action("fork", entry_id, options or {})

    async def navigate_tree(self, target_id: str, options: dict | None = None):
        return await self._runner._command_action("navigate_tree", target_id, options or {})

    async def switch_session(self, session_path: str, options: dict | None = None):
        return await self._runner._command_action("switch_session", session_path, options or {})

    async def reload(self) -> None:
        await self._runner._command_action("reload")


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
        self._shutdown_handler: Callable[[], None] | None = None
        self._abort_fn: Callable[[], None] | None = None
        self._command_handlers: dict[str, Callable] = {}

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
        self.runtime.set_action("set_active_tools", lambda names: self._set_active_tools(session, names))
        self.runtime.set_action("get_all_tools", lambda: self._get_all_tools(session))
        self.runtime.set_action("get_commands", self.get_registered_commands)
        self.apply_providers()

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
        return any(
            extension.handlers.get(event_type) for extension in self.extensions
        )

    async def emit_event(
        self, event_type: str, data: dict | None = None
    ) -> list[Any]:
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
            raise NotImplementedError(
                f"Command action '{name}' is not available in this mode"
            )
        result = handler(*args)
        if inspect.isawaitable(result):
            return await result
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
            invocation = command.name if counts[command.name] == 1 else f"{command.name}:{occurrence}"
            result.append(
                RegisteredCommand(
                    name=invocation,
                    description=command.description,
                    argument_hint=command.argument_hint,
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
        session._agent.state.tools = [
            by_name[name] for name in tool_names if name in by_name
        ]

    def _get_all_tools(self, session) -> list[dict]:
        if session is None:
            return []
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in session._agent.state.tools
        ]

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
            session.follow_up(text)
        elif options.get("deliverAs") == "steer":
            session.steer(text)
        else:
            import asyncio

            asyncio.create_task(self._send_prompt_text(session, text))

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
            import asyncio

            try:
                asyncio.get_running_loop().create_task(self.session.abort())
            except RuntimeError:
                pass

    def shutdown(self) -> None:
        if self._shutdown_handler is not None:
            self._shutdown_handler()

    async def shutdown_all(self) -> None:
        """卸载：分发 session_shutdown（后续可扩展 deactivate 钩子）。"""
        if self.has_handlers("session_shutdown"):
            await self.emit_event("session_shutdown")


__all__ = [
    "ExtensionRunner",
    "ExtensionContext",
    "ExtensionCommandContext",
]
