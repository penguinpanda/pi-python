"""Slash 命令系统（对齐 TS core/slash-commands.ts）。"""

from __future__ import annotations

import inspect
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
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
        open_tree_selector: Callable[[], None] | None = None,
        open_fork_selector: Callable[[], None] | None = None,
        copy_to_clipboard: Callable[[str], None] | None = None,
        auth_interaction=None,
        rebuild_session=None,
        reload_all: Callable[[], Any] | None = None,
    ) -> None:
        self.session = session
        self.model_runtime = model_runtime
        self.keybindings_manager = keybindings_manager
        self.slash_registry = slash_registry
        self._notify = notify or (lambda _message: None)
        self._exit_app = exit_app
        self._new_session = new_session
        self._open_model_selector = open_model_selector
        self._open_tree_selector = open_tree_selector
        self._open_fork_selector = open_fork_selector
        self._copy_to_clipboard = copy_to_clipboard or (lambda _text: None)
        self.auth_interaction = auth_interaction
        # 会话重建：fork / clone / resume / import 用（宿主注入）。
        self.rebuild_session = rebuild_session
        # 宿主级重载：/reload 用（TUI 注入）。
        self.reload_all = reload_all

    def notify(self, message: str) -> None:
        self._notify(message)

    def exit_app(self) -> None:
        if self._exit_app is not None:
            self._exit_app()

    def new_session(self) -> None:
        if self._new_session is not None:
            self._new_session()

    async def open_model_selector(self) -> None:
        if self._open_model_selector is not None:
            result = self._open_model_selector()
            if inspect.isawaitable(result):
                await result

    def open_tree_selector(self) -> None:
        if self._open_tree_selector is not None:
            self._open_tree_selector()

    def open_fork_selector(self) -> None:
        if self._open_fork_selector is not None:
            self._open_fork_selector()

    def copy_to_clipboard(self, text: str) -> None:
        self._copy_to_clipboard(text)

    @property
    def auth_store(self):
        runtime = self.model_runtime
        return getattr(runtime, "auth_store", None) if runtime is not None else None

    async def rebuild(self, manager):
        """重建并替换会话（返回新 AgentSession）。"""
        if self.rebuild_session is None:
            raise RuntimeError("Session rebuild is not available in this mode")
        result = self.rebuild_session(manager)
        return await result if inspect.isawaitable(result) else result

    async def reload(self) -> str:
        """触发宿主级 reload（/reload 用）。"""
        if self.reload_all is None:
            return "Reload is not available in this mode"
        result = self.reload_all()
        return await result if inspect.isawaitable(result) else result


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
        parts = _split_args(stripped[1:])
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
            await context.open_model_selector()
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
            # 对齐 TS handleCompactCommand：无可压缩内容时静默，不渲染消息。
            return ""
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
        return await context.reload()

    async def _copy(context: SlashContext, _args: str) -> str:
        text = context.session.get_last_assistant_text()
        if not text:
            return "No assistant message to copy"
        context.copy_to_clipboard(text)
        return "Copied last assistant message"

    async def _export(context: SlashContext, args: str) -> str:
        from ...export_html import export_session_to_html

        output = args.strip() or str(
            Path(context.session.cwd) / f"session-{context.session.session_id}.html"
        )
        path = export_session_to_html(context.session.session_manager, output)
        return f"Exported session to {path}"

    async def _tree(context: SlashContext, args: str) -> str:
        manager = context.session.session_manager
        target = args.strip()
        if target:
            if manager.get_entry(target) is None:
                return f"Entry not found: {target}"
            await context.session.navigate_to(target)
            return f"Navigated to {target}"
        # TUI 环境：打开树选择器弹层（对齐 TS TreeSelectorComponent）；
        # 无回调（RPC 等）时退回文本输出。
        if context._open_tree_selector is not None:
            context.open_tree_selector()
            return ""
        lines = _format_tree(manager.get_tree(), manager.get_leaf_id())
        return "\n".join(lines) if lines else "(empty session)"

    async def _fork(context: SlashContext, args: str) -> str:
        entry_id = args.strip()
        if not entry_id:
            # TUI 环境：打开树选择器选择要 fork 的消息（对齐 TS /fork 打开选择器）。
            if context._open_fork_selector is not None:
                context.open_fork_selector()
                return ""
            return "Usage: /fork <entryId>"
        manager = context.session.session_manager
        if manager.get_entry(entry_id) is None:
            return f"Entry not found: {entry_id}"
        forked = manager.fork(entry_id)
        new_session = await context.rebuild(forked)
        context.session = new_session
        return f"Forked at {entry_id} (session {new_session.session_id})"

    async def _clone(context: SlashContext, _args: str) -> str:
        leaf_id = context.session.session_manager.get_leaf_id()
        if leaf_id is None:
            return "No current entry to clone"
        forked = context.session.session_manager.fork(leaf_id)
        new_session = await context.rebuild(forked)
        context.session = new_session
        return f"Cloned to session {new_session.session_id}"

    async def _settings(context: SlashContext, args: str) -> str:
        from ..._config import (
            _deep_merge,
            _load_json,
            get_project_settings_path,
            get_settings_path,
            load_settings,
        )

        cwd = context.session.cwd
        arg = args.strip()
        if arg:
            if "=" not in arg:
                return "Usage: /settings [key=value]"
            key, value = arg.split("=", 1)
            try:
                parsed = json.loads(value.strip())
            except ValueError:
                parsed = value.strip()
            path = get_project_settings_path(cwd)
            data = _load_json(path)
            data[key.strip()] = parsed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return f"Saved {key.strip()} = {parsed} to {path}"
        merged = load_settings(cwd)
        return (
            f"Settings (global: {get_settings_path()}, project: {get_project_settings_path(cwd)}):\n"
            + json.dumps(merged, ensure_ascii=False, indent=2)
        )

    async def _scoped_models(context: SlashContext, args: str) -> str:
        args_str = args.strip()
        if args_str == "clear":
            context.session.set_scoped_models([])
            return "Scoped models cleared (all available are usable)"
        if not args_str:
            scoped = context.session.scoped_models
            if not scoped:
                return "No scoped models (all available are usable)"
            return "Scoped models:\n" + "\n".join(
                f"  {entry.model.provider}/{entry.model.id}" for entry in scoped
            )
        from ...model_resolver import resolve_model_scope

        patterns = [
            part
            for piece in shlex.split(args_str)
            for part in piece.split(",")
            if part
        ]
        scoped = await resolve_model_scope(patterns, context.model_runtime)
        context.session.set_scoped_models(scoped)
        return f"Scoped {len(scoped)} models"

    async def _login(context: SlashContext, args: str) -> str:
        from pi_ai.auth.oauth import builtin_oauth_providers

        providers = builtin_oauth_providers()
        available = ", ".join(provider_id for provider_id, _name, _flow in providers)
        provider_id = args.strip() or None
        if provider_id is None:
            return f"Usage: /login <provider>. Available: {available}"
        match = next(
            (provider for provider in providers if provider[0] == provider_id), None
        )
        if match is None:
            return f"Unknown provider: {provider_id}. Available: {available}"
        _pid, _name, flow = match
        if context.auth_store is None:
            return "Auth store not available"
        try:
            interaction = (
                context.auth_interaction
                if context.auth_interaction is not None
                else _TerminalAuthInteraction()
            )
            credential = await flow.login(interaction)
        except Exception as exc:
            return f"Login failed: {exc}"
        async def _set(_current):
            return credential

        await context.auth_store.modify(_pid, _set)
        return f"Logged in: {_name} ({_pid})"

    async def _logout(context: SlashContext, args: str) -> str:
        provider_id = args.strip()
        if not provider_id:
            return "Usage: /logout <provider>"
        await context.model_runtime.logout(provider_id)
        return f"Logged out: {provider_id}"

    async def _share(context: SlashContext, _args: str) -> str:
        import httpx

        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            return "No GITHUB_TOKEN environment variable set"
        text = _session_to_markdown(context.session)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.github.com/gists",
                    headers={
                        "Authorization": f"token {token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json={
                        "description": f"pi session {context.session.session_id}",
                        "public": False,
                        "files": {
                            f"session-{context.session.session_id}.md": {
                                "content": text
                            }
                        },
                    },
                )
        except Exception as exc:
            return f"Failed to share: {exc}"
        if response.status_code >= 300:
            return f"Failed to share: HTTP {response.status_code}"
        return f"Shared: {response.json().get('html_url')}"

    async def _import_session(context: SlashContext, args: str) -> str:
        path = args.strip()
        if not path:
            return "Usage: /import <session.jsonl>"
        from ..._session_manager import SessionManager

        manager = SessionManager.open(path, cwd_override=context.session.cwd)
        new_session = await context.rebuild(manager)
        context.session = new_session
        return f"Imported {manager.session_id} ({len(manager.get_entries())} entries)"

    async def _resume(context: SlashContext, args: str) -> str:
        from ..._config import get_sessions_dir
        from ..._session_manager import SessionManager

        path = args.strip()
        if path:
            manager = SessionManager.open(path, cwd_override=context.session.cwd)
            new_session = await context.rebuild(manager)
            context.session = new_session
            return f"Resumed {new_session.session_id}"
        infos = SessionManager.list_sessions(get_sessions_dir())
        if not infos:
            return "No saved sessions"
        lines = ["Saved sessions (use /resume <path>):"]
        for info in infos[:10]:
            lines.append(f"  {info.path}  ({info.session_id})")
        return "\n".join(lines)

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
        ("export", _export, "Export session (HTML/JSONL)", "[path]"),
        ("tree", _tree, "Navigate session tree", "<entryId>"),
        ("fork", _fork, "Create a new fork from a previous message", "<entryId>"),
        ("clone", _clone, "Duplicate the current session", ""),
        ("settings", _settings, "Show or edit settings", "[key=value]"),
        ("scoped-models", _scoped_models, "Enable/disable models for Ctrl+P cycling", "[pattern...]"),
        ("login", _login, "Configure provider authentication", "<provider>"),
        ("logout", _logout, "Remove provider authentication", "<provider>"),
        ("share", _share, "Share session as a secret GitHub gist", ""),
        ("import", _import_session, "Import and resume a session from a JSONL file", "<path>"),
        ("resume", _resume, "Resume a different session", "[path]"),
    ]
    for name, handler, description, hint in builtins:
        registry.register(name, handler, description=description, argument_hint=hint)


def _format_tree(nodes, leaf_id: str | None, depth: int = 0) -> list[str]:
    """把会话树渲染为缩进文本（leaf 标记 >）。"""
    lines: list[str] = []
    for node in nodes:
        marker = ">" if node.id == leaf_id else " "
        label = f" [{node.label}]" if node.label else ""
        entry_type = node.entry.get("type", "?") if node.entry is not None else "?"
        lines.append(f"{'  ' * depth}{marker} {node.id[:8]} {entry_type}{label}")
        lines.extend(_format_tree(node.children, leaf_id, depth + 1))
    return lines


def _split_args(args_string: str) -> list[str]:
    """切分参数：支持引号；保留反斜杠（Windows 路径不被转义破坏）。"""
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def _session_to_markdown(session) -> str:
    """把会话分支渲染为 Markdown（/share 用）。"""
    lines: list[str] = [f"# pi session {session.session_id}", ""]
    for entry in session.session_manager.get_branch():
        if entry.get("type") != "message":
            continue
        message = entry.get("message") or {}
        role = message.get("role", "agent")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = ""
        if text:
            lines.append(f"**{role}**: {text}")
            lines.append("")
    return "\n".join(lines)


class _TerminalAuthInteraction:
    """OAuth 登录的终端交互适配。"""

    signal = None

    async def prompt(self, prompt) -> str:
        if prompt.get("type") == "select":
            options = prompt.get("options") or []
            for index, option in enumerate(options, 1):
                print(f"  {index}. {option.get('label', '')}")
            while True:
                raw = input(f"Enter number (1-{len(options)}): ").strip()
                try:
                    return options[int(raw) - 1]["id"]
                except (ValueError, IndexError):
                    print("Invalid selection.")
        return input(f"{prompt.get('message', '')}: ")

    def notify(self, event) -> None:
        if event.get("type") == "auth_url":
            print(f"\nOpen this URL in your browser:\n{event['url']}")
        elif event.get("type") == "device_code":
            print(f"\nOpen {event.get('verificationUri', '')} and enter: {event.get('userCode', '')}")
        elif event.get("message"):
            print(event["message"])


__all__ = [
    "SlashCommand",
    "SlashContext",
    "SlashCommandRegistry",
    "register_builtin_commands",
]
