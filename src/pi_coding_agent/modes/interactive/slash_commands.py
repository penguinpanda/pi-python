"""Slash 命令系统（对齐 TS core/slash-commands.ts）。"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SlashCommand:
    """单个 slash 命令。"""

    name: str
    handler: Callable[["SlashContext", str], Any]
    description: str = ""
    argument_hint: str | None = None


class SlashContext:
    """slash 命令执行上下文（TUI 注入）。"""

    def __init__(
        self,
        *,
        session=None,
        model_runtime=None,
        keybindings_manager=None,
        slash_registry=None,
        notify: Callable[[str], None] | None = None,
        exit_app: Callable[[], None] | None = None,
        new_session: Callable[[], None] | None = None,
        open_model_selector: Callable[[], None] | None = None,
        copy_to_clipboard: Callable[[str], None] | None = None,
    ) -> None:
        self.session = session
        self.model_runtime = model_runtime
        self.keybindings_manager = keybindings_manager
        self.slash_registry = slash_registry
        self._notify = notify or (lambda _message: None)
        self._exit_app = exit_app
        self._new_session = new_session
        self._open_model_selector = open_model_selector
        self._copy_to_clipboard = copy_to_clipboard or (lambda _text: None)

    def notify(self, message: str) -> None:
        self._notify(message)

    def exit_app(self) -> None:
        if self._exit_app is not None:
            self._exit_app()

    def new_session(self) -> None:
        if self._new_session is not None:
            self._new_session()

    def open_model_selector(self) -> None:
        if self._open_model_selector is not None:
            self._open_model_selector()

    def copy_to_clipboard(self, text: str) -> None:
        self._copy_to_clipboard(text)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SlashCommandRegistry:
    """slash 命令注册表与解析执行。"""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(
        self,
        name: str,
        handler: Callable[[SlashContext, str], Any],
        *,
        description: str = "",
        argument_hint: str | None = None,
    ) -> None:
        self._commands[name] = SlashCommand(
            name=name,
            handler=handler,
            description=description,
            argument_hint=argument_hint,
        )

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def list(self) -> list[SlashCommand]:
        return list(self._commands.values())

    @staticmethod
    def parse(text: str) -> tuple[str | None, str]:
        """解析 `/name arg1 arg2` → (name, args)。"""
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None, ""
        try:
            parts = shlex.split(stripped[1:])
        except ValueError:
            parts = stripped[1:].split()
        if not parts:
            return "", ""
        return parts[0], " ".join(parts[1:])

    async def execute(self, text: str, context: SlashContext) -> bool:
        """识别并执行 slash 命令；返回 True 表示已处理。"""
        name, args = self.parse(text)
        if name is None:
            return False
        if name == "":
            context.notify("Usage: /<command> [args]")
            return True
        command = self._commands.get(name)
        if command is None:
            context.notify(f"Unknown command: /{name}. Use /help to list commands.")
            return True
        result = command.handler(context, args)
        if result is not None and hasattr(result, "__await__"):
            message = await result
        else:
            message = result
        if message:
            context.notify(message)
        return True


# ---------------------------------------------------------------------------
# 内置命令
# ---------------------------------------------------------------------------


def register_builtin_commands(registry: SlashCommandRegistry) -> None:
    """注册 pi 内置 slash 命令。"""

    async def _model(context: SlashContext, args: str) -> str:
        from ...model_resolver import resolve_cli_model

        if not args.strip():
            context.open_model_selector()
            return ""
        resolved = resolve_cli_model(
            cli_provider=None,
            cli_model=args.strip(),
            model_runtime=context.model_runtime,
        )
        if resolved.error:
            return f"Error: {resolved.error}"
        if resolved.model is None:
            return f'Model "{args}" not found.'
        await context.session.set_model(resolved.model)
        return f"Switched to {resolved.model.provider}/{resolved.model.id}"

    async def _name(context: SlashContext, args: str) -> str:
        name = args.strip()
        if not name:
            return "Usage: /name <name>"
        context.session.set_session_name(name)
        return f"Session name set to: {name}"

    async def _compact(context: SlashContext, args: str) -> str:
        result = await context.session.compact(args.strip() or None)
        if result is None:
            return "Nothing to compact"
        return "Session compacted"

    async def _new(context: SlashContext, _args: str) -> None:
        context.new_session()
        return None

    async def _quit(context: SlashContext, _args: str) -> None:
        context.exit_app()
        return None

    async def _help(context: SlashContext, _args: str) -> str:
        lines = ["Available commands:"]
        commands = (
            context.slash_registry.list()
            if context.slash_registry is not None
            else []
        )
        for command in sorted(commands, key=lambda c: c.name):
            hint = f" {command.argument_hint}" if command.argument_hint else ""
            lines.append(f"  /{command.name}{hint} — {command.description}")
        return "\n".join(lines)

    async def _hotkeys(context: SlashContext, _args: str) -> str:
        manager = context.keybindings_manager
        lines = ["Keybindings:"]
        for binding in manager.all_bindings():
            lines.append(f"  {binding.key:<16} {binding.description}")
        return "\n".join(lines)

    async def _session(context: SlashContext, _args: str) -> str:
        stats = context.session.get_session_stats()
        return (
            f"Session {stats['sessionId']}: "
            f"{stats['userMessages']} user / {stats['assistantMessages']} assistant, "
            f"{stats['totalMessages']} messages, cost ${stats['cost']:.6f}"
        )

    async def _reload(context: SlashContext, _args: str) -> str:
        context.notify("Reloaded keybindings, skills, prompts, and themes")
        return ""

    async def _copy(context: SlashContext, _args: str) -> str:
        text = context.session.get_last_assistant_text()
        if not text:
            return "No assistant message to copy"
        context.copy_to_clipboard(text)
        return "Copied last assistant message"

    def _not_implemented(name: str) -> Callable[[SlashContext, str], str]:
        async def _handler(_context: SlashContext, _args: str) -> str:
            return f"/{name} is not implemented yet"

        return _handler

    builtins: list[tuple[str, Callable, str, str | None]] = [
        ("model", _model, "Select model (opens selector UI)", "<provider/model>"),
        ("name", _name, "Set session display name", "<name>"),
        ("compact", _compact, "Manually compact the session context", "[instructions]"),
        ("new", _new, "Start a new session", ""),
        ("quit", _quit, "Quit pi", ""),
        ("help", _help, "List available commands", ""),
        ("hotkeys", _hotkeys, "Show all keyboard shortcuts", ""),
        ("session", _session, "Show session info and stats", ""),
        ("reload", _reload, "Reload keybindings, skills, prompts, and themes", ""),
        ("copy", _copy, "Copy last agent message to clipboard", ""),
        ("export", _not_implemented("export"), "Export session (HTML/JSONL)", "[path]"),
        ("tree", _not_implemented("tree"), "Navigate session tree", ""),
        ("fork", _not_implemented("fork"), "Create a new fork from a previous message", ""),
        ("clone", _not_implemented("clone"), "Duplicate the current session", ""),
        ("settings", _not_implemented("settings"), "Open settings menu", ""),
        ("scoped-models", _not_implemented("scoped-models"), "Enable/disable models for Ctrl+P cycling", ""),
        ("login", _not_implemented("login"), "Configure provider authentication", "<provider>"),
        ("logout", _not_implemented("logout"), "Remove provider authentication", ""),
        ("share", _not_implemented("share"), "Share session as a secret GitHub gist", ""),
        ("import", _not_implemented("import"), "Import and resume a session from a JSONL file", ""),
        ("resume", _not_implemented("resume"), "Resume a different session", ""),
    ]
    for name, handler, description, hint in builtins:
        registry.register(name, handler, description=description, argument_hint=hint)


__all__ = [
    "SlashCommand",
    "SlashContext",
    "SlashCommandRegistry",
    "register_builtin_commands",
]
