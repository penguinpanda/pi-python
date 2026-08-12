"""扩展类型定义（5.1）——对齐 TS core/extensions/types.ts 的子集。"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, cast


# ---------------------------------------------------------------------------
# 事件总线（扩展间通信）
# ---------------------------------------------------------------------------


class EventBus:
    """极简发布/订阅（对齐 TS core/event-bus.ts）。"""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> Callable[[], None]:
        self._listeners.setdefault(event, []).append(handler)

        def _unsubscribe() -> None:
            try:
                self._listeners[event].remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    def emit(self, event: str, data: Any = None) -> None:
        for handler in list(self._listeners.get(event, [])):
            try:
                handler(data)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UI 上下文
# ---------------------------------------------------------------------------


class UIContext(Protocol):
    """扩展可用的 UI 抽象（TUI/RPC/Print 各自实现）。"""

    async def select(
        self, title: str, options: list[str], timeout: float | None = None
    ) -> str | None: ...

    async def confirm(self, title: str, message: str, timeout: float | None = None) -> bool: ...

    async def input(
        self, title: str, placeholder: str | None = None, timeout: float | None = None
    ) -> str | None: ...

    def notify(self, message: str, notify_type: str | None = None) -> None: ...

    def set_status(self, key: str, text: str | None) -> None: ...

    def set_title(self, title: str) -> None: ...

    def set_editor_text(self, text: str) -> None: ...

    def set_footer(self, text: str | None) -> None: ...

    def set_header(self, text: str | None) -> None: ...

    def set_editor_component(self, component) -> None: ...

    def set_widget(
        self, key: str, lines: list[str] | None, options: dict | None = None
    ) -> None: ...

    def set_overlay(self, key: str, lines: list[str], options: dict | None = None) -> None: ...

    def set_overlay_component(self, key: str, component, options: dict | None = None) -> None: ...

    def set_overlay_renderer(self, key: str, renderer, options: dict | None = None) -> None: ...

    def set_hidden_thinking_label(self, label: str | None = None) -> None: ...

    def set_working_message(self, text: str | None = None) -> None: ...

    def set_working_visible(self, visible: bool) -> None: ...

    def set_working_indicator(self, options: dict | None = None) -> None: ...

    def paste_to_editor(self, text: str) -> None: ...

    def get_editor_text(self) -> str: ...

    async def editor(self, title: str, prefill: str = "") -> str | None: ...

    def set_theme(self, theme: str | None = None) -> None: ...

    async def custom(
        self,
        factory,
        *,
        overlay_options: dict | None = None,
        on_handle=None,
    ) -> Any: ...


class NoopUIContext:
    """Print 模式降级 UI（所有操作 no-op）。"""

    async def select(self, title, options, timeout=None):
        return None

    async def confirm(self, title, message, timeout=None):
        return False

    async def input(self, title, placeholder=None, timeout=None):
        return None

    def notify(self, message, notify_type=None):
        pass

    def set_status(self, key, text):
        pass

    def set_title(self, title):
        pass

    def set_editor_text(self, text):
        pass

    def set_footer(self, text):
        pass

    def set_header(self, text):
        pass

    def set_editor_component(self, component):
        pass

    def set_widget(self, key, lines, options=None):
        pass

    def set_overlay(self, key, lines, options=None):
        pass

    def set_overlay_component(self, key, component, options=None):
        pass

    def set_overlay_renderer(self, key, renderer, options=None):
        pass

    def set_hidden_thinking_label(self, label=None):
        pass

    def set_working_message(self, text=None):
        pass

    def set_working_visible(self, visible):
        pass

    def set_working_indicator(self, options=None):
        pass

    def paste_to_editor(self, text):
        pass

    def get_editor_text(self):
        return ""

    async def editor(self, title, prefill=""):
        return None

    def set_theme(self, theme=None):
        pass

    async def custom(self, factory, *, overlay_options=None, on_handle=None):
        return None


# ---------------------------------------------------------------------------
# 注册项
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolDefinition:
    """扩展注册的 LLM 工具。"""

    name: str
    description: str = ""
    prompt_snippet: str = ""
    prompt_guidelines: list[str] | None = None
    parameters: dict | None = None
    label: str = ""
    source_info: dict | None = None
    execute: Callable | None = None
    execution_mode: str = "parallel"
    render_call: Callable | None = None
    render_result: Callable | None = None


@dataclass(slots=True)
class RegisteredCommand:
    """扩展注册的 /command。"""

    name: str
    description: str = ""
    argument_hint: str | None = None
    get_argument_completions: Callable | None = None
    handler: Callable | None = None
    source_info: dict | None = None


@dataclass(slots=True)
class ExtensionShortcut:
    """扩展注册的键盘快捷键。"""

    shortcut: str
    description: str = ""
    handler: Callable | None = None
    extension_path: str = ""


@dataclass(slots=True)
class ExtensionFlag:
    """扩展注册的 CLI 标志。"""

    name: str
    description: str = ""
    type: str = "boolean"
    default: bool | str | None = None
    extension_path: str = ""


@dataclass(slots=True)
class ExtensionError:
    """扩展错误（事件分发失败等）。"""

    extension_path: str
    event: str
    error: str
    stack: str | None = None


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Extension:
    """已加载的扩展实例。"""

    path: str
    resolved_path: str
    source: str = "local"
    base_dir: str | None = None
    hidden: bool = False
    handlers: dict[str, list[Callable]] = field(default_factory=dict)
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    shortcuts: dict[str, ExtensionShortcut] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    message_renderers: dict[str, Callable] = field(default_factory=dict)
    tool_renderers: dict[str, Callable] = field(default_factory=dict)
    entry_renderers: dict[str, Callable] = field(default_factory=dict)
    markdown_transformers: list[Callable] = field(default_factory=list)
    providers: list[tuple[str, dict]] = field(default_factory=list)
    autocomplete: list[Callable] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 运行时（flag 值 + 动作实现）
# ---------------------------------------------------------------------------


class ExtensionRuntime:
    """共享运行时：flag 值与动作实现（注册期为存根，绑定后替换）。"""

    def __init__(self) -> None:
        self.flag_values: dict[str, bool | str | None] = {}
        self._actions: dict[str, Callable] = {}

    def set_action(self, name: str, fn: Callable) -> None:
        self._actions[name] = fn

    def get_action(self, name: str):
        return self._actions.get(name)


def _not_initialized(action: str) -> Callable:
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"Extension action '{action}' not initialized. "
            "Action methods cannot be called during extension loading."
        )

    return _raise


# ---------------------------------------------------------------------------
# ExtensionAPI（扩展模块收到的 pi 对象）
# ---------------------------------------------------------------------------


class ExtensionAPI:
    """扩展可用的 API：注册方法写入 extension；动作方法委托 runtime。"""

    def __init__(
        self,
        extension: Extension,
        runtime: ExtensionRuntime,
        *,
        cwd: str,
        event_bus: EventBus | None = None,
    ) -> None:
        self._extension = extension
        self._runtime = runtime
        self._cwd = cwd
        self.events = event_bus or EventBus()

    # -- 事件 --

    def on(self, event_type: str, handler: Callable) -> None:
        self._extension.handlers.setdefault(event_type, []).append(handler)

    # -- 注册 --

    def register_tool(self, tool: ToolDefinition | dict) -> None:
        definition = tool if isinstance(tool, ToolDefinition) else ToolDefinition(**tool)
        if definition.source_info is None:
            definition.source_info = {
                "source": self._extension.source,
                "path": self._extension.path,
            }
        self._extension.tools[definition.name] = definition

    def register_command(self, name: str, options: dict | None = None) -> None:
        options = dict(options or {})
        if "getArgumentCompletions" in options:
            options["get_argument_completions"] = options.pop("getArgumentCompletions")
        options["name"] = name
        options["source_info"] = {"source": self._extension.source, "path": self._extension.path}
        self._extension.commands[name] = RegisteredCommand(**options)

    def register_shortcut(self, shortcut: str, options: dict | None = None) -> None:
        options = dict(options or {})
        options["shortcut"] = shortcut
        options["extension_path"] = self._extension.path
        self._extension.shortcuts[shortcut] = ExtensionShortcut(**options)

    def register_flag(self, name: str, options: dict | None = None) -> None:
        options = dict(options or {})
        options["name"] = name
        options["extension_path"] = self._extension.path
        flag = ExtensionFlag(**options)
        self._extension.flags[name] = flag
        if flag.default is not None and name not in self._runtime.flag_values:
            self._runtime.flag_values[name] = flag.default

    def get_flag(self, name: str) -> bool | str | None:
        if name not in self._extension.flags:
            return None
        return self._runtime.flag_values.get(name)

    def set_flag_value(self, name: str, value: bool | str | None) -> None:
        """写入扩展 flag 的运行时值（CLI 解析后调用）。"""
        if name not in self._extension.flags:
            return
        self._runtime.flag_values[name] = value

    def register_message_renderer(self, custom_type: str, renderer: Callable) -> None:
        self._extension.message_renderers[custom_type] = renderer

    def register_tool_renderer(self, tool_name: str, renderer: Callable) -> None:
        """注册内置/自定义工具结果的 TUI 渲染器（返回字符串）。"""
        self._extension.tool_renderers[tool_name] = renderer

    def register_entry_renderer(self, custom_type: str, renderer: Callable) -> None:
        self._extension.entry_renderers[custom_type] = renderer

    def register_markdown_transformer(self, transformer: Callable) -> None:
        self._extension.markdown_transformers.append(transformer)

    def register_autocomplete(self, provider: Callable) -> None:
        self._extension.autocomplete.append(provider)

    def register_provider(self, name: str, config: dict) -> None:
        self._extension.providers.append((name, dict(config)))

    def unregister_provider(self, name: str) -> None:
        self._extension.providers = [
            (provider_name, config)
            for provider_name, config in self._extension.providers
            if provider_name != name
        ]

    # -- 动作（委托 runtime） --

    def set_model(self, model):
        action = self._runtime.get_action("set_model") or _not_initialized("set_model")
        return action(model)

    def get_thinking_level(self):
        action = self._runtime.get_action("get_thinking_level") or _not_initialized(
            "get_thinking_level"
        )
        return action()

    def set_thinking_level(self, level: str) -> None:
        action = self._runtime.get_action("set_thinking_level") or _not_initialized(
            "set_thinking_level"
        )
        action(level)

    def set_session_name(self, name: str) -> None:
        action = self._runtime.get_action("set_session_name") or _not_initialized(
            "set_session_name"
        )
        action(name)

    def get_session_name(self):
        action = self._runtime.get_action("get_session_name")
        return action() if action is not None else None

    def send_user_message(self, content: str, options: dict | None = None) -> None:
        action = self._runtime.get_action("send_user_message") or _not_initialized(
            "send_user_message"
        )
        action(content, options or {})

    def send_message(self, content: Any, options: dict | None = None) -> None:
        """发送自定义消息（role=custom，写入会话树并进入上下文）。"""
        action = self._runtime.get_action("send_message") or _not_initialized("send_message")
        action(content, options or {})

    def append_entry(self, custom_type: str, data: Any = None) -> None:
        """追加自定义会话条目（custom 类型）。"""
        action = self._runtime.get_action("append_entry") or _not_initialized("append_entry")
        action(custom_type, data)

    def set_label(self, entry_id: str, label: str | None) -> None:
        """给会话条目设置 label（/tree 导航用）。"""
        action = self._runtime.get_action("set_label") or _not_initialized("set_label")
        action(entry_id, label)

    def get_active_tools(self):
        action = self._runtime.get_action("get_active_tools")
        return action() if action is not None else []

    def set_active_tools(self, tool_names: list[str]) -> None:
        action = self._runtime.get_action("set_active_tools") or _not_initialized(
            "set_active_tools"
        )
        action(tool_names)

    def get_all_tools(self):
        action = self._runtime.get_action("get_all_tools")
        return action() if action is not None else []

    def get_commands(self):
        action = self._runtime.get_action("get_commands")
        return action() if action is not None else []

    async def exec(self, command: str, args: list[str] | None = None, options: dict | None = None):
        from pi_agent import PythonExecutionEnv, ShellExecOptions

        full = command if not args else f"{command} {shlex.join(args)}"
        env = PythonExecutionEnv(cwd=(options or {}).get("cwd") or self._cwd)
        ok, result = await env.exec(
            full,
            ShellExecOptions(timeout=int((options or {}).get("timeout") or 120)),
        )
        if not ok:
            raise result
        return {
            "output": (cast(Any, result).stdout or "") + (cast(Any, result).stderr or ""),
            "exit_code": cast(Any, result).exit_code,
            "canceled": False,
        }


__all__ = [
    "EventBus",
    "UIContext",
    "NoopUIContext",
    "ToolDefinition",
    "RegisteredCommand",
    "ExtensionShortcut",
    "ExtensionFlag",
    "ExtensionError",
    "Extension",
    "ExtensionRuntime",
    "ExtensionAPI",
]
