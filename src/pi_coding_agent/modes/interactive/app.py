"""pi TUI 主应用（对齐 TS modes/interactive/）。"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap

from ..._config import get_sessions_dir
from ..._session import AgentSession
from ...model_runtime import ModelRuntime
from ...extensions import ExtensionRunner
from ...extensions.registry import ExtensionRegistry
from pi_tui.clipboard_image import ClipboardImage
from pi_tui.components import (
    MessageEntry,
    PiChatContainer,
    PiEditor,
    PiFooter,
    PiHeader,
    PiStatusBar,
)
from pi_tui.keybindings import KeybindingsManager
from pi_tui.selectors import (
    ModelSelector,
    SessionPicker,
    TextInputDialog,
    TreeSelector,
)
from .slash_commands import (
    SlashContext,
    SlashCommandRegistry,
    register_builtin_commands,
)
from pi_tui.theme import ThemeLoader


class _TuiAuthInteraction:
    """TUI 内 OAuth 交互：URL/设备码发到聊天区，需要粘贴时弹输入框。

    对齐 TS 的 showLoginDialog：不阻塞 Textual 事件循环（不能用 input()）。
    """

    def __init__(self, app: "PiTuiApp") -> None:
        self.app = app
        self.signal = None
        self._last_auth_url: str | None = None

    def notify(self, event: dict) -> None:
        if event.get("type") == "auth_url":
            self._last_auth_url = event["url"]
            # 自动复制完整 URL，避免终端折行截断导致链接不完整。
            self.app.copy_to_clipboard(event["url"])
            self.app._slash_notify(
                "请在浏览器完成授权（URL 已复制到剪贴板），"
                "然后把最终重定向 URL 粘贴回来：\n" + event["url"]
            )
        elif event.get("type") == "device_code":
            self.app.copy_to_clipboard(event.get("userCode", ""))
            self.app._slash_notify(
                f"请打开 {event['verificationUri']} 并输入代码："
                f"{event['userCode']}（代码已复制到剪贴板）"
            )
        else:
            message = event.get("message")
            if message:
                self.app._slash_notify(message)

    async def prompt(self, prompt) -> str | None:
        if prompt.get("type") == "select":
            return None
        message = prompt.get("message", "")
        if prompt.get("type") == "manual_code" and self._last_auth_url:
            # 对话框里同时显示授权 URL，方便复制（用户手动打开浏览器）。
            message = (
                "使用步骤：\n"
                "1. 授权 URL 已复制到剪贴板，直接粘贴到浏览器地址栏打开并完成授权\n"
                "2. 完成后浏览器会跳转到本地回调地址（页面打不开属正常）\n"
                "3. 把地址栏里的完整 URL 复制到下方输入框，按 Enter 提交\n\n"
                f"授权 URL：\n{self._last_auth_url}\n\n{message}"
            )
        return await self.app._await_text_input(
            message,
            placeholder=prompt.get("placeholder", ""),
        )


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

ModelSelector, SessionPicker, TreeSelector, TextInputDialog {
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
        extension_loader=None,
    ) -> None:
        self._keybindings = keybindings_manager or KeybindingsManager()
        self._settings = settings if settings is not None else {}
        if self._settings:
            self._keybindings.load_from_settings(self._settings)
        self._theme_loader = theme_loader or ThemeLoader()
        self._theme_name = theme_name
        self._theme = self._theme_loader.resolve(theme_name)
        self._extension_loader = extension_loader

        # 实例级 BINDINGS / CSS：必须在 super().__init__() 之前设置。
        self.BINDINGS = [
            Binding(binding.key, binding.action, binding.description)
            for binding in self._keybindings.all_bindings()
        ]
        self.CSS = _build_css(self._theme.colors)
        super().__init__()
        # Textual 8 的运行时绑定来自 node._bindings（类级 BINDINGS 构建）；
        # 实例级 self.BINDINGS 需显式重建映射，否则快捷键不会分发。
        self._bindings = BindingsMap(self.BINDINGS)

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
            notify=self._slash_notify,
            exit_app=self.exit,
            new_session=self._handle_new_session,
            open_model_selector=self._open_model_selector,
            open_tree_selector=self._open_tree_selector,
            open_fork_selector=self._open_fork_selector,
            copy_to_clipboard=self._copy_to_clipboard,
            auth_interaction=_TuiAuthInteraction(self),
            reload_all=self._reload_all,
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
    def _header(self) -> PiHeader:
        return self.query_one("#pi-header", PiHeader)

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

    def _copy_to_clipboard(self, text: str) -> None:
        """复制到剪贴板：优先 OSC 52（终端处理，可穿过 docker exec），失败回退系统工具。"""
        try:
            self.copy_to_clipboard(text)
        except Exception:
            _copy_text(text)

    def _slash_notify(self, message: str) -> None:
        """slash 命令输出：状态栏 + 聊天区。

        多行输出（如 /tree）在单行状态栏会被裁剪，看起来像没反应，
        因此同时渲染为聊天区的 System 消息。
        """
        self._set_status(message)
        # 转义方括号，避免树文本里的 [label] 被 Textual 标记解析。
        escaped = message.replace("[", r"\[")
        self._chat.add_message_agent({"role": "system", "content": escaped})
        # add_message_agent 会 scroll_end 到最底部，长消息只露出最后一行；
        # 这里把整条 System 消息滚进视口（等 mount 完成后再滚）。
        entries = self._chat.query(MessageEntry)
        if entries:
            entry = entries[-1]

            def _scroll_to_entry() -> None:
                try:
                    self._chat.scroll_visible(entry, animate=False)
                except Exception:
                    pass

            self.call_after_refresh(_scroll_to_entry)

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

    def on_pi_editor_exit_requested(self, _message) -> None:
        """编辑器为空时 ctrl+d → 退出。"""
        self.action_exit()

    def on_pi_editor_copy_requested(self, _message) -> None:
        """ctrl+x → 复制最后一条 assistant 消息（对齐 TS）。"""
        self.action_copy_last_message()

    def on_pi_editor_cycle_thinking_requested(self, _message) -> None:
        """shift+tab → 循环 thinking 级别（对齐 TS）。"""
        self.action_cycle_thinking()

    def action_external_editor(self) -> None:
        """ctrl+g：用外部编辑器编辑当前输入（对齐 TS app.editor.external）。"""
        import os
        import shlex
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(self._editor.text)
            path = handle.name
        editor = (
            os.environ.get("EDITOR")
            or os.environ.get("VISUAL")
            or ("notepad" if os.name == "nt" else "vi")
        )
        try:
            subprocess.run([*shlex.split(editor), path], check=False)
            with open(path, "r", encoding="utf-8") as handle:
                updated = handle.read()
            self._editor.text = updated
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

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

    async def _await_text_input(self, message: str, placeholder: str = "") -> str | None:
        """弹输入框并等待结果（OAuth 登录等 TUI 交互用，不阻塞事件循环）。"""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.push_screen(
            TextInputDialog(message, placeholder),
            callback=lambda value: future.set_result(value),
        )
        return await future

    def _open_tree_selector(self) -> None:
        manager = self._session.session_manager
        tree = manager.get_tree()
        if not tree:
            self._notify("(empty session)")
            return
        self.push_screen(
            TreeSelector(tree, leaf_id=manager.get_leaf_id()),
            callback=self._on_tree_selected,
        )

    def _on_tree_selected(self, entry_id) -> None:
        if entry_id is None:
            return
        self._run_task(self._navigate_tree(entry_id))

    async def _navigate_tree(self, entry_id: str) -> None:
        try:
            await self._session.navigate_to(entry_id)
            self._notify(f"Navigated to {entry_id[:8]}")
        except Exception as exc:
            self._notify(f"Navigate failed: {exc}")

    def _open_fork_selector(self) -> None:
        manager = self._session.session_manager
        tree = manager.get_tree()
        if not tree:
            self._notify("(empty session)")
            return
        self.push_screen(
            TreeSelector(tree, leaf_id=manager.get_leaf_id()),
            callback=self._on_fork_selected,
        )

    def _on_fork_selected(self, entry_id) -> None:
        if entry_id is None:
            return
        self._run_task(self._fork_from_entry(entry_id))

    async def _fork_from_entry(self, entry_id: str) -> None:
        manager = self._session.session_manager
        if manager.get_entry(entry_id) is None:
            self._notify(f"Entry not found: {entry_id}")
            return
        try:
            forked = manager.fork(entry_id)
            new_session = await self._apply_rebuilt_session(forked)
            self._slash_context.session = new_session
            self._notify(f"Forked at {entry_id[:8]} (session {new_session.session_id})")
        except Exception as exc:
            self._notify(f"Fork failed: {exc}")

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

    async def _reload_all(self) -> str:
        """/reload：重新加载技能、提示模板、扩展、快捷键与主题。"""
        details: list[str] = []
        session = self._session

        # 1. 技能 / 提示模板（加载器原地重扫）。
        skill_loader = session.skill_loader
        if skill_loader is not None:
            try:
                result = skill_loader.reload()
                details.append(f"{len(result.skills)} skills")
            except Exception as exc:
                details.append(f"skills failed: {exc}")
        template_loader = session.template_loader
        if template_loader is not None:
            try:
                templates = template_loader.reload()
                details.append(f"{len(templates)} prompts")
            except Exception as exc:
                details.append(f"prompts failed: {exc}")

        # 2. 扩展：重扫磁盘 → 新 runner → 替换会话绑定。
        new_runner = None
        if self._extension_loader is not None:
            old_runner = session.extension_runner
            if old_runner is not None:
                try:
                    await old_runner.shutdown_all()
                except Exception:
                    pass
            try:
                result = await self._extension_loader.load()
                new_runner = ExtensionRunner(
                    result.extensions,
                    runtime=result.runtime,
                    cwd=session.cwd,
                    model_runtime=self._model_runtime,
                )
                session.set_extension_runner(new_runner)
                details.append(f"{len(result.extensions)} extensions")
            except Exception as exc:
                details.append(f"extensions failed: {exc}")
        elif session.extension_runner is not None:
            # 无 loader（测试 / 嵌入场景）：保留现有 runner，仅重放注册。
            new_runner = session.extension_runner

        # 3. 快捷键 + 扩展命令：重建注册表与 BINDINGS。
        try:
            registry = SlashCommandRegistry()
            register_builtin_commands(registry)
            self._keybindings.reset()
            self._keybindings.load_from_settings(self._settings)
            if new_runner is not None:
                ExtensionRegistry(
                    new_runner,
                    slash_registry=registry,
                    keybindings_manager=self._keybindings,
                ).apply()
            self._slash_registry = registry
            self._slash_context.slash_registry = registry
            self.BINDINGS = [
                Binding(binding.key, binding.action, binding.description)
                for binding in self._keybindings.all_bindings()
            ]
            self._bindings = BindingsMap(self.BINDINGS)
            self.refresh_bindings()
            self._header.refresh_hints()
            details.append("keybindings refreshed")
        except Exception as exc:
            details.append(f"keybindings failed: {exc}")

        # 4. 主题：重新解析并刷新 CSS。
        try:
            self._theme = self._theme_loader.resolve(self._theme_name)
            self.CSS = _build_css(self._theme.colors)
            self.refresh_css()
            details.append(f"theme {self._theme.name}")
        except Exception as exc:
            details.append(f"theme failed: {exc}")

        return "Reloaded: " + "; ".join(details)

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
        self._copy_to_clipboard(text)
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
