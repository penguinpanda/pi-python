"""pi TUI 主应用（对齐 TS modes/interactive/）。"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding

from ..._config import get_sessions_dir
from ..._session import AgentSession
from ...model_runtime import ModelRuntime
from ...extensions.registry import ExtensionRegistry
from pi_tui.clipboard_image import ClipboardImage
from pi_tui.components import (
    PiChatContainer,
    PiEditor,
    PiFooter,
    PiHeader,
    PiStatusBar,
)
from pi_tui.keybindings import KeybindingsManager
from pi_tui.selectors import ModelSelector, SessionPicker
from .slash_commands import (
    SlashContext,
    SlashCommandRegistry,
    register_builtin_commands,
)
from pi_tui.theme import ThemeLoader


_CSS_TEMPLATE = """
Screen {
    layout: vertical;
    background: __PI_BG__;
    color: __PI_TEXT__;
}

#pi-header {
    height: 1;
    background: __PI_BGPANEL__;
    color: __PI_TEXTALT__;
}

#pi-chat {
    height: 1fr;
    background: __PI_BG__;
    border: round __PI_BORDERINACTIVE__;
    padding: 0 1;
    scrollbar-size: 1 1;
}

MessageEntry {
    color: __PI_TEXT__;
    margin: 0 0 0 0;
}

#pi-status {
    height: 1;
    background: __PI_BGTOOLBAR__;
    color: __PI_TEXTSYSTEM__;
}

#pi-editor {
    height: 6;
    background: __PI_BGUSERINPUT__;
    color: __PI_TEXT__;
    border: round __PI_BORDERACTIVE__;
}

#pi-footer {
    height: 1;
    background: __PI_BGTOOLBAR__;
    color: __PI_TEXTDIM__;
}

.selector-title {
    color: __PI_ACCENT__;
    text-style: bold;
    margin: 1 2;
}

.selector-input {
    margin: 0 2 1 2;
}

ModelSelector, SessionPicker {
    background: __PI_BGPANEL__;
    color: __PI_TEXT__;
    border: round __PI_BORDERACTIVE__;
    height: 60%;
    width: 80%;
}
"""


def _build_css(colors: dict[str, str]) -> str:
    """把 __PI_<KEY>__ token 替换为主题色。"""
    css = _CSS_TEMPLATE
    for key, value in colors.items():
        css = css.replace(f"__PI_{key.upper()}__", value)
    return css


class PiTuiApp(App):
    """pi 编码代理 TUI。"""

    def __init__(
        self,
        session: AgentSession,
        model_runtime: ModelRuntime,
        *,
        keybindings_manager: KeybindingsManager | None = None,
        theme_loader: ThemeLoader | None = None,
        theme_name: str | None = "auto",
        session_factory: Callable[[], AgentSession] | None = None,
        resume_factory: Callable[[str], AgentSession] | None = None,
        session_rebuilder=None,
        settings: dict | None = None,
    ) -> None:
        self._keybindings = keybindings_manager or KeybindingsManager()
        if settings:
            self._keybindings.load_from_settings(settings)
        self._theme_loader = theme_loader or ThemeLoader()
        self._theme = self._theme_loader.resolve(theme_name)

        # 实例级 BINDINGS / CSS：必须在 super().__init__() 之前设置。
        self.BINDINGS = [
            Binding(binding.key, binding.action, binding.description)
            for binding in self._keybindings.all_bindings()
        ]
        self.CSS = _build_css(self._theme.colors)
        super().__init__()

        self._session = session
        self._model_runtime = model_runtime
        self._session_factory = session_factory
        self._resume_factory = resume_factory
        self._session_rebuilder = session_rebuilder
        self._unsubscribe: Callable[[], None] | None = None
        self._show_tools = True
        self._show_thinking = True
        self._tasks: set[asyncio.Task] = set()

        self._slash_registry = SlashCommandRegistry()
        register_builtin_commands(self._slash_registry)
        # 扩展命令 / 快捷键注入 slash 注册表与键位表。
        if session.extension_runner is not None:
            ExtensionRegistry(
                session.extension_runner,
                slash_registry=self._slash_registry,
                keybindings_manager=self._keybindings,
            ).apply()
        self._slash_context = SlashContext(
            session=session,
            model_runtime=model_runtime,
            keybindings_manager=self._keybindings,
            slash_registry=self._slash_registry,
            notify=self._notify,
            exit_app=self.exit,
            new_session=self._handle_new_session,
            open_model_selector=self._open_model_selector,
            copy_to_clipboard=_copy_text,
        )
        self._slash_context.rebuild_session = self._apply_rebuilt_session

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield PiHeader(self._keybindings, id="pi-header")
        yield PiChatContainer(id="pi-chat")
        yield PiStatusBar("Idle", id="pi-status")
        yield PiEditor(id="pi-editor")
        yield PiFooter("", id="pi-footer")

    def on_mount(self) -> None:
        self._bind_session()
        for message in self._session.get_messages():
            self._chat.add_message_agent(message)
        self._update_footer()
        self._editor.focus()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for task in list(self._tasks):
            task.cancel()

    def _bind_session(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._unsubscribe = self._session.subscribe(self._on_session_event)

    # ------------------------------------------------------------------
    # 组件快捷访问
    # ------------------------------------------------------------------

    @property
    def _chat(self) -> PiChatContainer:
        return self.query_one("#pi-chat", PiChatContainer)

    @property
    def _editor(self) -> PiEditor:
        return self.query_one("#pi-editor", PiEditor)

    @property
    def _status(self) -> PiStatusBar:
        return self.query_one("#pi-status", PiStatusBar)

    @property
    def _footer(self) -> PiFooter:
        return self.query_one("#pi-footer", PiFooter)

    # ------------------------------------------------------------------
    # 会话事件 → UI
    # ------------------------------------------------------------------

    def _on_session_event(self, event: dict) -> None:
        try:
            event_type = event.get("type")
            if event_type == "message_end":
                message = event.get("message")
                if message is not None:
                    self._chat.add_message_agent(message)
                self._update_footer()
            elif event_type == "agent_settled":
                self._set_status("Idle")
                self._update_footer()
            elif event_type == "compaction_start":
                self._set_status("Compacting")
            elif event_type in ("compaction_end",):
                self._set_status("Idle")
            elif event_type in ("model_changed", "thinking_level_changed"):
                self._update_footer()
            elif event_type == "agent_start":
                self._set_status("Working")
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        self._status.update(text)

    def _notify(self, message: str) -> None:
        self._set_status(message)

    def _update_footer(self) -> None:
        model = self._session.model
        model_label = f"{model.provider}/{model.id}" if model is not None else "—"
        self._footer.update_info(
            model=model_label,
            thinking=self._session.thinking_level,
            message_count=len(self._session.get_messages()),
            session_name=self._session.session_name,
        )

    # ------------------------------------------------------------------
    # 编辑器
    # ------------------------------------------------------------------

    def on_pi_editor_submitted(self, message: PiEditor.Submitted) -> None:
        text = message.text
        if text.startswith("/"):
            self._run_task(self._exec_slash(text))
        else:
            self._run_task(self._send_prompt(text))

    async def _send_prompt(self, text: str) -> None:
        self._set_status("Working")
        try:
            await self._session.prompt(text)
        except Exception as exc:
            self._notify(f"Prompt failed: {exc}")

    async def _exec_slash(self, text: str) -> None:
        # Phase 4：/skill:name 与 /templateName 先经会话管道展开。
        expanded = self._session.expand_prompt(text)
        if expanded != text:
            await self._send_prompt(expanded)
            return
        try:
            await self._slash_registry.execute(text, self._slash_context)
        except Exception as exc:
            self._notify(f"Command failed: {exc}")
        self._update_footer()

    # ------------------------------------------------------------------
    # 快捷键 actions
    # ------------------------------------------------------------------

    def action_interrupt(self) -> None:
        self._set_status("Aborting")
        self._run_task(self._session.abort())

    def action_clear(self) -> None:
        self._editor.clear()

    def action_exit(self) -> None:
        if not self._editor.text.strip():
            self.exit()
        else:
            self._editor.clear()

    def action_cycle_thinking(self) -> None:
        level = self._session.cycle_thinking_level()
        if level is None:
            self._notify("Model does not support thinking")
        self._update_footer()

    def action_cycle_model_forward(self) -> None:
        self._run_task(self._cycle_model(1))

    def action_cycle_model_backward(self) -> None:
        self._run_task(self._cycle_model(-1))

    async def _cycle_model(self, direction: int) -> None:
        try:
            result = await self._session.cycle_model(direction)
            if result is None:
                self._notify("Only one model available")
        except Exception as exc:
            self._notify(f"Cycle model failed: {exc}")
        self._update_footer()

    def action_select_model(self) -> None:
        self._run_task(self._open_model_selector())

    async def _open_model_selector(self) -> None:
        models = await self._model_runtime.get_available()
        if not models:
            self._notify("No models available")
            return
        self.push_screen(
            ModelSelector(models, current=self._session.model),
            callback=self._on_model_selected,
        )

    def _on_model_selected(self, result) -> None:
        if result is not None:
            self._run_task(self._apply_selected_model(result))

    async def _apply_selected_model(self, model) -> None:
        try:
            await self._session.set_model(model)
        except Exception as exc:
            self._notify(f"Set model failed: {exc}")
        self._update_footer()

    def action_toggle_tools(self) -> None:
        self._show_tools = not self._show_tools
        self._rerender_chat()

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._rerender_chat()

    def _rerender_chat(self) -> None:
        self._chat.set_visibility(
            show_tools=self._show_tools, show_thinking=self._show_thinking
        )
        self._chat.clear_messages()
        for message in self._session.get_messages():
            self._chat.add_message_agent(message)

    def action_follow_up(self) -> None:
        text = self._editor.text.strip()
        if not text:
            return
        self._editor.clear()
        self._session.follow_up(text)
        self._set_status("Follow-up queued")

    def action_dequeue(self) -> None:
        self._session._agent.clear_all_queues()
        self._set_status("Queued messages cleared")

    def action_paste_image(self) -> None:
        self._run_task(self._paste_image())

    async def _paste_image(self) -> None:
        data = await ClipboardImage.read()
        if data is None:
            self._notify("No image in clipboard")
            return
        try:
            processed = await asyncio.to_thread(ClipboardImage.process, data)
        except Exception as exc:
            self._notify(f"Image processing failed: {exc}")
            return
        self._pending_image = processed
        self._set_status(f"Image pasted ({len(processed)} bytes); send a prompt to attach")

    def action_new_session(self) -> None:
        self._handle_new_session()

    def _handle_new_session(self) -> None:
        if self._session_factory is None:
            self._notify("New session not available")
            return
        self._run_task(self._create_new_session())

    async def _create_new_session(self) -> None:
        try:
            result = self._session_factory()
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            self._notify(f"New session failed: {exc}")
            return
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._chat.clear_messages()
        self._update_footer()
        self._set_status("New session")

    async def _apply_rebuilt_session(self, manager):
        """按给定 SessionManager 重建会话并替换（fork/clone/resume/import 用）。"""
        if self._session_rebuilder is None:
            raise RuntimeError("Session rebuild is not available")
        result = self._session_rebuilder(manager)
        new_session = await result if inspect.isawaitable(result) else result
        await self._replace_session(new_session)
        return new_session

    async def _replace_session(self, new_session: AgentSession) -> None:
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._chat.clear_messages()
        for message in new_session.get_messages():
            self._chat.add_message_agent(message)
        self._update_footer()
        self._set_status(f"Session {new_session.session_id}")

    def action_resume_session(self) -> None:
        self._run_task(self._resume_session())

    async def _resume_session(self) -> None:
        sessions = _list_sessions()
        if not sessions:
            self._notify("No saved sessions")
            return
        self.push_screen(
            SessionPicker(sessions),
            callback=self._on_session_selected,
        )

    def _on_session_selected(self, path) -> None:
        if path is None or self._resume_factory is None:
            return
        self._run_task(self._apply_resumed_session(path))

    async def _apply_resumed_session(self, path: str) -> None:
        try:
            result = self._resume_factory(path)
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Resume failed: {exc}")
            return
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._chat.clear_messages()
        for message in new_session.get_messages():
            self._chat.add_message_agent(message)
        self._update_footer()
        self._set_status(f"Resumed {new_session.session_id}")

    def action_copy_last_message(self) -> None:
        text = self._session.get_last_assistant_text()
        if not text:
            self._notify("No assistant message to copy")
            return
        _copy_text(text)
        self._notify("Copied last assistant message")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _run_task(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _list_sessions() -> list[dict[str, Any]]:
    """按修改时间倒序列出会话文件。"""
    sessions_dir = get_sessions_dir()
    if not sessions_dir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for path in sessions_dir.glob("*.jsonl"):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        session_id = path.stem
        try:
            with open(path, "r", encoding="utf-8") as handle:
                first = handle.readline().strip()
            header = json.loads(first)
            if isinstance(header, dict) and header.get("id"):
                session_id = header["id"]
        except (OSError, json.JSONDecodeError):
            pass
        results.append(
            {"path": str(path), "session_id": session_id, "modified": modified}
        )
    results.sort(key=lambda entry: entry["modified"], reverse=True)
    return results


def _copy_text(text: str) -> None:
    """尽力而为的剪贴板文本写入（失败静默）。"""
    try:
        if sys.platform == "win32":
            import subprocess

            subprocess.run(
                ["clip"], input=text.encode("utf-16-le") + b"\x00\x00", check=False
            )
        elif sys.platform == "darwin":
            import subprocess

            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
        else:
            import subprocess

            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=False,
            )
    except Exception:
        pass


async def run_tui_mode(
    session: AgentSession,
    model_runtime: ModelRuntime,
    **kwargs,
) -> int:
    """运行 TUI（供 CLI --mode tui 使用）。"""
    app = PiTuiApp(session, model_runtime, **kwargs)
    await app.run_async()
    return 0


__all__ = ["PiTuiApp", "run_tui_mode", "_list_sessions"]
