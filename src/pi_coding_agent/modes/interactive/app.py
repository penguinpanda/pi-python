"""pi TUI 主应用（引擎版，对齐 TS modes/interactive/）。"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

from rich.style import Style

from pi_agent import AgentEvent
from pi_tui.clipboard_image import ClipboardImage
from pi_tui.components import (
    BashExecutionEntry,
    MessageEntry,
    ToolExecutionEntry,
    PiChatContainer,
    PiEditor,
    PiFooter,
    PiHeader,
    PiStatusBar,
)
from pi_tui.engine import App, FakeTerminal, Terminal
from pi_tui.engine.keys import KeyEvent
from pi_tui.engine.widgets import Editor, Static
from pi_tui.keybindings import KeybindingsManager
from pi_tui.overlay import OverlayHandle
from pi_tui.selectors import (
    ExtensionSelector,
    ModelSelector,
    OAuthSelector,
    ScopedModelsSelector,
    SessionPicker,
    SettingsSelector,
    TextInputDialog,
    ThinkingSelector,
    TreeSelector,
    TrustSelector,
)
from pi_tui.theme import BUILTIN_THEMES, Theme, ThemeLoader

from ..._config import get_agent_dir, get_sessions_dir
from ..._session import AgentSession
from ..._session_manager import SessionManager, SessionTreeNode
from ..._session_manager_v4 import SessionManagerLike
from ...extensions import ExtensionRunner
from ...extensions.registry import ExtensionRegistry
from ...model_runtime import ModelRuntime
from ...system_prompt import load_project_context_files
from ...tools._ensure_tool import ensure_tool, get_tool_path
from ...tools.render_utils import get_text_output, shorten_path
from .autocomplete import create_interactive_autocomplete_provider
from .slash_commands import SlashContext, SlashCommandRegistry, register_builtin_commands


_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# autocomplete 请求编排（对齐 TS Editor 的 debounce + AbortController + requestId）：
# 连续输入停止 DEFAULT_AUTOCOMPLETE_DEBOUNCE_SECONDS 秒后才发起请求，
# 期间新输入会取消旧请求与旧 fd 子进程，旧请求结果按序号丢弃。
DEFAULT_AUTOCOMPLETE_DEBOUNCE_SECONDS = 0.15


class _TuiAuthInteraction:
    """TUI 内 OAuth 交互：URL/设备码发到聊天区，需要粘贴时弹输入框。"""

    def __init__(self, app: "PiTuiApp") -> None:
        self.app = app
        self.signal = None
        self._last_auth_url: str | None = None

    def notify(self, event: dict) -> None:
        if event.get("type") == "auth_url":
            self._last_auth_url = event["url"]
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


def _theme_style(theme: Theme, bg_key: str, fg_key: str) -> Style:
    """主题色 → Rich Style。"""
    return Style(
        bgcolor=theme.colors.get(bg_key, theme.colors["bg"]),
        color=theme.colors.get(fg_key, theme.colors["text"]),
    )


def _fg_style(theme: Theme, fg_key: str) -> Style:
    """仅前景色（对齐 TS：header/footer/status/输入框不涂背景）。"""
    return Style(color=theme.colors.get(fg_key, theme.colors["text"]))


class PiTuiApp(App):
    """pi 编码代理 TUI（引擎版）。"""

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
        settings_manager=None,
        trust_manager=None,
        project_trusted: bool = False,
        needs_trust_decision: bool = False,
        no_context_files: bool = False,
        startup_resources: dict | None = None,
        terminal: Terminal | FakeTerminal | None = None,
        size: tuple[int, int] = (80, 24),
        ui_mode: str | None = None,
    ) -> None:
        self._keybindings = keybindings_manager or KeybindingsManager()
        self._settings = settings if settings is not None else {}
        if self._settings:
            self._keybindings.load_from_settings(self._settings)
        self._settings_manager = settings_manager
        self._theme_loader = theme_loader or ThemeLoader()
        self._theme_name = theme_name
        self._theme = self._theme_loader.resolve(theme_name)
        self._extension_loader = extension_loader
        self._trust_manager = trust_manager
        self._project_trusted = project_trusted
        self._needs_trust_decision = needs_trust_decision
        self._no_context_files = no_context_files
        self._startup_resources = startup_resources

        self._ui_mode = ui_mode or str((settings or {}).get("uiMode", "regular"))
        super().__init__(
            keybindings=self._keybindings,
            terminal=terminal,
            size=size,
            ui_mode=self._ui_mode,
        )

        self._session = session
        self._model_runtime = model_runtime
        self._session_factory = session_factory
        self._resume_factory = resume_factory
        self._session_rebuilder = session_rebuilder
        self._unsubscribe: Callable[[], None] | None = None
        self._show_tools = True
        self._show_thinking = True
        if self._settings_manager is not None:
            self._show_thinking = not self._settings_manager.get_hide_thinking_block()
        self._tools_expanded = False
        self._chat_theme_colors: dict = {}
        self._rendered_summary_ids: set[str] = set()
        self._custom_editor: PiEditor | None = None
        self._widget_above: dict[str, str] = {}
        self._widget_below: dict[str, str] = {}
        self._overlay_dialog_callbacks: dict[str, Callable[[Any], None] | None] = {}
        self._overlay_renderers: dict[str, Callable[[int, int], list[str]]] = {}
        self._completion_items: list[dict] = []
        self._completion_index = 0
        self._completion_prefix = ""
        self._completion_kind = "text"
        # autocomplete 编排状态（debounce / abort / 请求序号）。
        self._autocomplete_debounce_task: asyncio.Task | None = None
        self._autocomplete_request_task: asyncio.Task | None = None
        self._autocomplete_request_id = 0
        self._hidden_thinking_label = "Thinking"
        self._working_message = "Working"
        self._working_visible = True
        self._stream_entry: MessageEntry | None = None
        self._tool_entries: dict[str, ToolExecutionEntry] = {}
        self._pending_image: bytes | None = None

        # 布局组件（on_mount 挂载）：对齐 TS transcript + dock 顺序。
        self._header = PiHeader(self._keybindings, height="auto", id="pi-header")
        self._resources = Static("", height="auto", id="pi-resources")
        self._chat = PiChatContainer(height="1fr", id="pi-chat")
        self._pending_messages = Static("", height="auto", id="pi-pending-messages")
        self._widgets_above = Static("", height="auto", id="pi-widgets-above")
        self._status = PiStatusBar("Idle", height=1, id="pi-status")
        self._editor_widget: PiEditor | None = None
        self._widgets_below = Static("", height="auto", id="pi-widgets-below")
        self._footer = PiFooter("", height=1, id="pi-footer")
        self._status_animated = False
        self._status_frame = 0
        self._status_base = "Idle"
        self._follow_up_queue: list[str] = []
        terminal_settings = (self._settings or {}).get("terminal") or {}
        self._show_images = bool(terminal_settings.get("showImages", True))
        width_cells = terminal_settings.get("imageWidthCells")
        self._image_width_cells = max(1, int(width_cells)) if width_cells else None
        self._show_terminal_progress = bool(terminal_settings.get("showTerminalProgress", False))
        self._quiet_startup = bool((settings or {}).get("quietStartup", False))
        self._autocomplete_max_visible = max(
            3, int((settings or {}).get("autocompleteMaxVisible", 5) or 5)
        )
        self.open_url = self._open_external_url

        self._slash_registry = SlashCommandRegistry()
        register_builtin_commands(self._slash_registry)
        # 扩展命令 / 快捷键注入 slash 注册表与键位表。
        if session.extension_runner is not None:
            extension_registry = ExtensionRegistry(
                session.extension_runner,
                slash_registry=self._slash_registry,
                keybindings_manager=self._keybindings,
            )
            extension_registry.apply()
            from .ui_context import TuiUIContext

            session.extension_runner.bind(ui_context=TuiUIContext(self))
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
            open_trust_selector=self._open_trust_selector,
            open_settings_selector=self._open_settings_selector,
            open_thinking_selector=self._open_thinking_selector,
            open_oauth_selector=self._open_oauth_selector,
            open_scoped_models_selector=self._open_scoped_models_selector,
            open_extensions_selector=self._open_extensions_selector,
            open_input_selector=self._open_input_selector,
            copy_to_clipboard=self._copy_to_clipboard,
            auth_interaction=_TuiAuthInteraction(self),
            reload_all=self._reload_all,
            trust_manager=trust_manager,
        )
        self._slash_context.rebuild_session = self._apply_rebuilt_session
        self._autocomplete_provider = self._build_autocomplete_provider()

    def _build_autocomplete_provider(self):
        """按当前 registry / 资源重建统一补全 provider。"""
        return create_interactive_autocomplete_provider(
            slash_registry=self._slash_registry,
            template_loader=self._session.template_loader,
            extension_runner=self._session.extension_runner,
            skill_loader=self._session.skill_loader,
            settings_manager=self._settings_manager,
            model_runtime=self._model_runtime,
            session=self._session,
            base_path=self._session.cwd,
            fd_path=get_tool_path("fd"),
        )

    @property
    def _editor(self) -> PiEditor:
        """当前编辑器：扩展替换后返回自定义组件。"""
        editor = self._custom_editor if self._custom_editor is not None else self._editor_widget
        assert editor is not None
        return editor

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        padding_x = int(self._settings.get("editorPaddingX", 0) or 0)
        self._editor_widget = PiEditor(height=6, id="pi-editor", border=True, padding_x=padding_x)
        # 布局选项对齐 TS fullscreenLayoutRoot：chat 弹性占满，dock 按自然高度，
        # 固定组件可 shrink 且有最小高度（editor 3 / footer 1）。
        # fullscreen：chat 占满剩余空间（basis 0 + grow 1）；
        # regular：chat 按内容自然高度展开，避免文档被压缩成 1 行。
        chat_basis: int | str = 0 if self._ui_mode == "fullscreen" else "auto"
        chat_grow = 1 if self._ui_mode == "fullscreen" else 0
        self.screen.mount(self._header, basis="auto", grow=0, shrink=1, min_size=0)
        self.screen.mount(self._resources, basis="auto", grow=0, shrink=1, min_size=0)
        self.screen.mount(self._chat, basis=chat_basis, grow=chat_grow, shrink=1, min_size=1)
        self.screen.mount(self._pending_messages, basis="auto", grow=0, shrink=1, min_size=0)
        self.screen.mount(self._status, basis=1, grow=0, shrink=1, min_size=0)
        self.screen.mount(self._widgets_above, basis="auto", grow=0, shrink=1, min_size=0)
        self.screen.mount(self._editor_widget, basis=6, grow=0, shrink=1, min_size=3)
        self.screen.mount(self._widgets_below, basis="auto", grow=0, shrink=1, min_size=0)
        self.screen.mount(self._footer, basis=1, grow=0, shrink=1, min_size=1)
        self._chat.set_image_options(
            show_images=self._show_images,
            image_width_cells=self._image_width_cells,
        )
        self._apply_theme()
        self._bind_session()
        for message in self._session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))
        self._render_missed_summaries()
        self._update_footer()
        self._editor.focus()
        self._show_startup_resources_hint()
        self._run_task(self._ensure_fd_for_autocomplete())
        # 启动时对未定信任项目提示（对齐 TS 启动 trust 选择器）。
        if self._needs_trust_decision:
            self._open_trust_selector()

    async def _ensure_fd_for_autocomplete(self) -> None:
        """后台确保 fd 可用，成功后重建补全 provider（对齐 TS ensureTool）。"""
        await ensure_tool("fd", silent=True)
        if get_tool_path("fd") is not None:
            self._autocomplete_provider = self._build_autocomplete_provider()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        # 退出时取消进行中的 autocomplete 编排任务（debounce + 请求）。
        if self._autocomplete_debounce_task is not None:
            self._autocomplete_debounce_task.cancel()
            self._autocomplete_debounce_task = None
        if self._autocomplete_request_task is not None:
            self._autocomplete_request_task.cancel()
            self._autocomplete_request_task = None

    def _bind_session(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._chat.set_renderers(
            custom_renderer=self._extension_custom_message_renderer,
            markdown_transformers=self._extension_markdown_transformers,
            tool_renderer=self._extension_tool_renderer,
        )
        self._unsubscribe = self._session.subscribe(
            cast(Callable[[AgentEvent], None], self._on_session_event)
        )

    def _apply_theme(self) -> None:
        theme = self._theme
        # 对齐 TS：终端背景透出，header/footer/status/输入框只设前景色。
        self.screen.base_style = _fg_style(theme, "text")
        for widget in (
            self._header,
            self._resources,
            self._chat,
            self._pending_messages,
            self._widgets_above,
            self._status,
            self._editor_widget,
            self._widgets_below,
            self._footer,
        ):
            if widget is None:
                continue
            if widget is self._header:
                widget.base_style = _fg_style(theme, "textAlt")
            elif widget is self._status:
                widget.base_style = _fg_style(theme, "textSystem")
            elif widget is self._editor_widget:
                # 正文用 muted textAlt（对齐 TS：编辑器文字不染色、终端默认灰），
                # 光标保持反色块，避免整行看起来都是光标色；不涂输入框底色。
                widget.base_style = _fg_style(theme, "textAlt")
                widget.border_style = Style(
                    color=theme.colors.get("border", theme.colors["textAlt"]),
                )
            elif widget is self._footer:
                widget.base_style = _fg_style(theme, "textDim")
            else:
                widget.base_style = _fg_style(theme, "text")
        self._chat_theme_colors = {
            "heading": theme.colors.get("accent"),
            "code_fg": theme.colors.get("text"),
            "code_bg": theme.colors.get("bgPanel", theme.colors.get("bgAlt")),
            "thinking": theme.colors.get("thinkingText"),
            "toolPendingBg": theme.colors.get("toolPendingBg"),
            "toolSuccessBg": theme.colors.get("toolSuccessBg"),
            "toolErrorBg": theme.colors.get("toolErrorBg"),
            "toolTitle": theme.colors.get("toolTitle"),
            "toolOutput": theme.colors.get("toolOutput"),
            "bashMode": theme.colors.get("bashMode"),
            "dim": theme.colors.get("dim"),
        }
        self._chat.set_theme_colors(self._chat_theme_colors)
        self.request_render()

    # ------------------------------------------------------------------
    # 会话事件 → UI
    # ------------------------------------------------------------------

    def _on_session_event(self, event: dict) -> None:
        try:
            event_type = event.get("type")
            if event_type == "message_start":
                # 对齐 TS：只有 assistant 消息才创建流式占位；user/custom
                # 由 message_end 统一追加，避免出现空的 Speaking 残留条目。
                role = (event.get("message") or {}).get("role")
                if role == "assistant":
                    self._begin_stream()
            elif event_type == "message_update":
                self._update_stream(event.get("message"))
            elif event_type == "message_end":
                self._finish_stream()
                message = event.get("message")
                if message is not None:
                    self._render_tool_calls(message)
                    self._chat.add_message_agent(
                        message, skip_tool_calls=(message.get("role") == "assistant")
                    )
                self._update_footer()
            elif event_type == "agent_settled":
                self._finish_stream()
                self.terminal.set_progress(False)
                self._set_status("Idle", animated=False)
                self._update_footer()
            elif event_type == "compaction_start":
                self._set_status("Compacting", animated=True)
            elif event_type in ("compaction_end",):
                self._set_status("Idle", animated=False)
            elif event_type in ("model_changed", "thinking_level_changed"):
                self._update_footer()
            elif event_type == "queue_update":
                self._update_pending_messages(event)
            elif event_type == "session_info_changed":
                self._update_terminal_title()
            elif event_type == "agent_start":
                self._follow_up_queue.clear()
                self.terminal.set_progress(self._show_terminal_progress)
                if self._working_visible:
                    self._set_status(self._working_message, animated=True)
                else:
                    self._set_status("Idle", animated=False)
            elif event_type == "skill_invocation":
                skill = event.get("skill", "")
                self._chat.add_message_agent(
                    {
                        "role": "skillInvocation",
                        "content": f"Invoked skill: {skill}",
                    }
                )
            elif event_type == "tool_execution_start":
                self._on_tool_execution_start(event)
            elif event_type == "tool_execution_update":
                self._on_tool_execution_update(event)
            elif event_type == "tool_execution_end":
                self._on_tool_execution_end(event)
        except Exception:
            pass

    def _begin_stream(self) -> None:
        """消息开始：挂一个流式占位条目。"""
        if self._stream_entry is not None:
            return
        entry = MessageEntry("Assistant", "")
        entry.set_speaking(True)
        self._stream_entry = entry
        self._chat.mount(entry)
        self._chat.scroll_end()

    def _update_stream(self, message) -> None:
        """消息增量：把 partial 快照渲染到流式条目。"""
        if not isinstance(message, dict):
            return
        if self._stream_entry is None:
            self._begin_stream()
        if self._stream_entry is None:
            return
        parts: list[str] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and block.get("text"):
                parts.append(str(block.get("text", "")))
            elif block_type == "thinking" and self._show_thinking and block.get("thinking"):
                parts.append(str(block.get("thinking", "")))
            elif block_type == "toolCall":
                parts.append(f"{block.get('name', 'tool')}({block.get('arguments', {})})")
        self._render_tool_calls(message)
        self._stream_entry.set_text("\n\n".join(parts))
        self._chat.scroll_end()

    def _finish_stream(self) -> None:
        """消息结束：移除流式占位（最终消息由 message_end 正常追加）。"""
        entry = self._stream_entry
        self._stream_entry = None
        if entry is not None and entry.is_mounted:
            try:
                entry.remove()
            except Exception:
                pass

    def _render_tool_calls(self, message) -> None:
        """从 assistant 消息内容创建/更新工具执行条目（对齐 TS ToolExecutionComponent）。"""
        if not isinstance(message, dict):
            return
        content = message.get("content") or []
        if not isinstance(content, list):
            return
        created = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            entry, was_created = self._ensure_tool_entry(
                str(block.get("name", "tool")),
                str(block.get("id") or block.get("index") or len(self._tool_entries)),
                block.get("arguments", {}),
            )
            created = created or was_created
            result = block.get("result")
            if isinstance(result, dict):
                output = self._tool_result_text(result)
                entry.set_result(
                    output,
                    is_error=bool(result.get("isError") or result.get("is_error")),
                    result=result,
                )
        if created:
            self._chat.scroll_end()

    def _ensure_tool_entry(
        self,
        tool_name: str,
        call_id: str,
        arguments: Any,
    ) -> tuple[ToolExecutionEntry, bool]:
        """查找或创建工具执行条目（tool_execution_* 事件与消息块共用）。"""
        entry = self._tool_entries.get(call_id)
        if entry is not None:
            entry.update_arguments(arguments)
            return entry, False
        render_call = None
        render_result = None
        runner = self._session.extension_runner
        if runner is not None:
            definition = runner.get_tool_definition(tool_name)
            if definition is not None:
                render_call = definition.render_call
                render_result = definition.render_result
        from .ui_context import ThemeFacade

        entry = ToolExecutionEntry(
            tool_name,
            call_id,
            arguments,
            render_call=render_call,
            render_result=render_result,
            render_theme=ThemeFacade(self._theme),
            theme_colors=self._chat_theme_colors,
            image_width=self._image_width_cells,
        )
        entry.set_expanded(self._tools_expanded)
        self._tool_entries[call_id] = entry
        self._chat.mount(entry)
        return entry, True

    def _result_to_dict(self, result: Any) -> dict | None:
        """工具结果（dict / AgentToolResult）统一为 content 块字典。"""
        if isinstance(result, dict):
            return result
        if result is None:
            return None
        content = getattr(result, "content", None)
        return {"content": list(content or [])}

    def _tool_result_text(self, result: Any) -> str:
        payload = self._result_to_dict(result)
        if payload is None:
            return ""
        output = get_text_output(payload, self._show_images)
        if output:
            return output
        raw = payload.get("content") or payload.get("text") or payload.get("details")
        if raw is None:
            return ""
        return str(raw)

    def _on_tool_execution_start(self, event: dict) -> None:
        """工具开始执行：创建条目并挂进聊天（对齐 TS tool_execution_start）。"""
        entry, created = self._ensure_tool_entry(
            str(event.get("tool_name", "tool")),
            str(event.get("tool_call_id")),
            event.get("args", {}),
        )
        if created:
            self._chat.scroll_end()

    def _on_tool_execution_update(self, event: dict) -> None:
        """工具流式局部结果（对齐 TS tool_execution_update）。"""
        call_id = str(event.get("tool_call_id"))
        entry = self._tool_entries.get(call_id)
        if entry is None:
            return
        result = event.get("result") or event.get("partial_result")
        output = self._tool_result_text(result)
        entry.set_partial_result(output, result=result)

    def _on_tool_execution_end(self, event: dict) -> None:
        """工具执行完成（对齐 TS tool_execution_end）。"""
        call_id = str(event.get("tool_call_id"))
        entry = self._tool_entries.get(call_id)
        if entry is None:
            return
        result = event.get("result")
        output = self._tool_result_text(result)
        entry.set_result(
            output,
            is_error=bool(event.get("is_error")),
            result=result,
        )

    def _extension_custom_message_renderer(self, message):
        runner = self._session.extension_runner
        if runner is None:
            return None
        renderer = runner.get_message_renderer(str(message.get("customType", "")))
        if renderer is None:
            return None
        result = renderer(message)
        if asyncio.iscoroutine(result):
            return None
        return result

    def _extension_markdown_transformers(self):
        runner = self._session.extension_runner
        if runner is None:
            return []
        return runner.get_markdown_transformers()

    def _extension_tool_renderer(self, message):
        runner = self._session.extension_runner
        if runner is None:
            return None
        renderer = runner.get_tool_renderer(str(message.get("tool_name", "")))
        if renderer is None:
            return None
        result = renderer(message)
        if asyncio.iscoroutine(result):
            return None
        return result

    def _replace_editor(self, component) -> None:
        """用扩展提供的编辑器组件替换 PiEditor。"""
        from pi_tui import PiEditor as PiEditorType

        if not isinstance(component, PiEditorType):
            raise TypeError("set_editor_component requires a PiEditor subclass")
        if self._editor_widget is not None:
            self._editor_widget.visible = False
        component.id = f"pi-editor-{id(component):x}"
        component.height_spec = 6
        if self._editor_widget is not None and getattr(self._editor_widget, "border_style", None):
            component.border_style = self._editor_widget.border_style
        self.screen.mount(component, basis=6, grow=0, shrink=1, min_size=3)
        self._custom_editor = component
        component.focus()

    def _set_status(self, text: str, animated: bool = False) -> None:
        """状态栏文本；animated=True 时显示 spinner（对齐 TS 状态指示器）。"""
        self._status_base = text
        self._status_animated = bool(animated)
        if self._status is not None:
            if animated:
                spinner = _SPINNER_FRAMES[self._status_frame % len(_SPINNER_FRAMES)]
                self._status.update(f"{spinner} {text}")
            else:
                self._status.update(text)

    def _set_widget(self, key: str, lines: list[str], options: dict | None = None) -> None:
        """编辑器上方（默认）或下方显示多行组件。"""
        below = (options or {}).get("placement") == "belowEditor"
        target = self._widget_below if below else self._widget_above
        if lines:
            target[key] = "\n".join(lines)
        else:
            target.pop(key, None)
        widget = self._widgets_below if below else self._widgets_above
        if widget is not None:
            widget.update("\n".join(target.values()))

    def _set_hidden_thinking_label(self, label: str | None = None) -> None:
        self._hidden_thinking_label = label or "Thinking"
        if self._chat is not None:
            self._chat.set_hidden_thinking_label(self._hidden_thinking_label)

    def _set_working_message(self, text: str | None = None) -> None:
        self._working_message = text or "Working"

    def _update_pending_messages(self, event: dict | None = None) -> None:
        """渲染 follow-up / steer 队列（对齐 TS pendingMessagesContainer）。"""
        lines: list[str] = []
        if event is not None:
            groups = (
                ("Steer", event.get("steer") or []),
                ("Follow-up", event.get("follow_up") or []),
                ("Next", event.get("next_turn") or []),
            )
            for label, items in groups:
                texts = [
                    str(getattr(item, "content", item))
                    if not isinstance(item, dict)
                    else str(item.get("content", item))
                    for item in items
                ]
                texts = [text for text in texts if text]
                if texts:
                    lines.append(f"[dim]{label}: {', '.join(texts)}[/dim]")
        else:
            if self._follow_up_queue:
                lines.append(f"[dim]Follow-up: {', '.join(self._follow_up_queue)}[/dim]")
        self._pending_messages.update("\n".join(lines))

    def _update_terminal_title(self) -> None:
        """对齐 TS updateTerminalTitle：pi - 会话名 - cwd 目录名。"""
        try:
            session_name = self._session.session_name
            cwd = str(self._session.cwd or "")
            basename = os.path.basename(cwd.rstrip("/\\")) or cwd
            if session_name:
                self.set_title(f"pi - {session_name} - {basename}")
            else:
                self.set_title(f"pi - {basename}")
        except Exception:
            pass

    def on_tick(self) -> None:
        """状态 spinner 动画（由 App 每帧 tick 驱动）。"""
        if self._status_animated:
            self._status_frame += 1
            if self._status is not None:
                spinner = _SPINNER_FRAMES[self._status_frame % len(_SPINNER_FRAMES)]
                self._status.update(f"{spinner} {self._status_base}")

    def _set_theme(self, theme: str | None = None) -> None:
        try:
            self._theme = self._theme_loader.resolve(theme or self._theme_name)
            self._apply_theme()
        except Exception:
            pass

    def _set_overlay(
        self,
        key: str,
        lines: list[str],
        options: dict | None = None,
    ) -> OverlayHandle | None:
        """显示 / 更新浮层；空列表移除。"""
        if not lines:
            self._overlay_renderers.pop(key, None)
            self._overlay_manager.remove(key)
            return None
        self._overlay_renderers.pop(key, None)
        handle = self._overlay_manager.show(key, list(lines), options or {})
        self._overlay_manager.reposition(key)
        return handle

    def _set_overlay_component(
        self,
        key: str,
        component,
        options: dict | None = None,
    ) -> OverlayHandle | None:
        """用任意组件作为 overlay；None 移除。"""
        if component is None:
            self._overlay_renderers.pop(key, None)
            self._overlay_manager.remove(key)
            return None
        component.app = self
        self._overlay_renderers.pop(key, None)
        handle = self._overlay_manager.show_component(key, component, options or {})
        self._overlay_manager.reposition(key)
        self._overlay_manager.ensure_focus(key)
        return handle

    def _set_overlay_renderer(
        self,
        key: str,
        renderer,
        options: dict | None = None,
    ) -> OverlayHandle | None:
        """用渲染回调生成 overlay 内容：fn(width, height) -> list[str]。"""
        if renderer is None:
            self._overlay_renderers.pop(key, None)
            self._overlay_manager.remove(key)
            return None
        self._overlay_renderers[key] = renderer
        handle = self._overlay_manager.show(key, [], options or {})
        self._render_overlay_renderer(key)
        return handle

    def _render_overlay_renderer(self, key: str) -> None:
        renderer = self._overlay_renderers.get(key)
        entry = self._overlay_manager.get(key)
        if renderer is None or entry is None:
            return
        width, height = self._overlay_manager.term_size
        try:
            lines = renderer(width, height)
        except Exception:
            lines = ["(renderer error)"]
        if not isinstance(lines, list):
            lines = ["(renderer error)"]
        entry.widget.update_content([str(line) for line in lines])
        self._overlay_manager.reposition(key)

    def push_screen(self, component, callback=None, wait_for_dismiss=False, *, mode=None) -> Any:
        """选择器挂进 overlay 层（选择器即 overlay）。"""
        component.app = self
        key = f"dialog-{id(component):x}"
        self._overlay_dialog_callbacks[key] = callback
        self._overlay_manager.show_component(
            key,
            component,
            {"anchor": "center", "width": "80%", "maxHeight": "60%"},
        )
        self._overlay_manager.reposition(key)
        self._overlay_manager.ensure_focus(key)
        return None

    def _close_overlay_dialog(self, component, value=None) -> None:
        """对话框 dismiss：移除 overlay 并回调结果。"""
        entry = self._overlay_manager.entry_for_widget(component)
        if entry is None:
            # 组件未挂到 overlay 树上（引擎 overlay 根直接持有 component）。
            for candidate in self._overlay_manager.entries.values():
                if candidate.widget is component or candidate.widget.component() is component:
                    entry = candidate
                    break
        if entry is None:
            return
        key = entry.key
        callback = self._overlay_dialog_callbacks.pop(key, None)
        self._overlay_manager.remove(key)
        if callback is not None:
            callback(value)

    def _notify(self, message: str) -> None:
        self._set_status(message)

    def _open_external_url(self, url: str) -> None:
        """打开 OSC8 链接（对齐 TS openUrl → openBrowser）。"""
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            self.copy_to_clipboard(text)
        except Exception:
            _copy_text(text)

    def _slash_notify(self, message: str) -> None:
        """slash 命令输出：状态栏 + 聊天区。"""
        self._set_status(message)
        self._chat.add_message_agent({"role": "system", "content": message})
        entries = self._chat.query(MessageEntry)
        if entries:
            self._chat.scroll_to_widget(entries[-1])

    def _show_startup_resources_hint(self) -> None:
        """启动提示：已加载资源汇总。"""
        sections: list[tuple[str, list[str]]] = []
        resources = self._startup_resources or {}

        if not self._no_context_files:
            context_files = resources.get("context_files")
            if context_files is None:
                context_files = load_project_context_files(self._session.cwd, get_agent_dir())
            if context_files:
                sections.append(
                    (
                        "Context",
                        [
                            _format_context_path(entry["path"], self._session.cwd)
                            for entry in context_files
                        ],
                    )
                )

        skills = [str(item["name"]) for item in resources.get("skills", []) if item.get("name")]
        if skills:
            sections.append(("Skills", skills))
        prompts = [str(item["name"]) for item in resources.get("prompts", []) if item.get("name")]
        if prompts:
            sections.append(("Prompts", prompts))
        extensions = [
            str(item["name"]) for item in resources.get("extensions", []) if item.get("name")
        ]
        if extensions:
            sections.append(("Extensions", extensions))
        custom_themes = [
            name for name in self._theme_loader.available() if name not in BUILTIN_THEMES
        ]
        if custom_themes:
            sections.append(("Themes", custom_themes))

        if self._quiet_startup or not sections:
            self._resources.update("")
            return
        text = "\n".join(f"[{title}]\n  {', '.join(items)}" for title, items in sections)
        self._set_status("Resources loaded")
        self._resources.update(text)

    def _update_footer(self) -> None:
        model = self._session.model
        model_label = f"{model.provider}/{model.id}" if model is not None else "—"
        self._footer.update_info(
            model=model_label,
            thinking=self._session.thinking_level,
            message_count=len(self._session.get_messages()),
            session_name=self._session.session_name,
        )
        self._update_terminal_title()

    # ------------------------------------------------------------------
    # 编辑器
    # ------------------------------------------------------------------

    def on_pi_editor_submitted(self, message: PiEditor.Submitted) -> None:
        text = message.text
        self._hide_slash_completion()
        if text.startswith("/"):
            self._run_task(self._exec_slash(text))
        elif text.startswith("!"):
            self._editor.add_to_history(text)
            self._run_task(self._exec_bash(text))
        else:
            self._editor.add_to_history(text)
            self._run_task(self._send_prompt(text))

    def on_pi_editor_autocomplete_requested(
        self,
        message: PiEditor.AutocompleteRequested,
    ) -> None:
        """Tab / 自动触发：debounce + abort + 请求序号编排后走非模态内联补全。

        对齐 TS Editor：每次触发取消上一 debounce 与进行中的请求（AbortController），
        debounce 到期后发起新请求（autocompleteRequestId），旧请求结果按序号丢弃，
        不覆盖新输入。
        """
        editor = message.editor
        if self._autocomplete_debounce_task is not None:
            self._autocomplete_debounce_task.cancel()
            self._autocomplete_debounce_task = None
        if self._autocomplete_request_task is not None:
            self._autocomplete_request_task.cancel()
            self._autocomplete_request_task = None
        self._autocomplete_request_id += 1
        request_id = self._autocomplete_request_id
        text_snapshot = editor.text
        cursor_snapshot = self._editor_cursor(editor)

        async def _debounced() -> None:
            await asyncio.sleep(DEFAULT_AUTOCOMPLETE_DEBOUNCE_SECONDS)
            if request_id != self._autocomplete_request_id:
                return
            self._autocomplete_request_task = self._run_task(
                self._fetch_autocomplete_suggestions(
                    editor, text_snapshot, cursor_snapshot, request_id
                )
            )

        self._autocomplete_debounce_task = self._run_task(_debounced())

    async def _fetch_autocomplete_suggestions(
        self,
        editor: Editor,
        text: str,
        cursor: int,
        request_id: int,
    ) -> None:
        try:
            suggestions = await self._autocomplete_provider.get_suggestions(
                text,
                force=True,
                cursor=cursor,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            suggestions = None
        finally:
            if self._autocomplete_request_task is asyncio.current_task():
                self._autocomplete_request_task = None
        if request_id != self._autocomplete_request_id or editor.text != text:
            # 已有更新请求（序号前进）或编辑器内容已变：丢弃过期结果。
            return
        self._render_autocomplete_suggestions(suggestions)

    def _render_autocomplete_suggestions(self, suggestions) -> None:
        if suggestions is None or not suggestions.items:
            self._hide_slash_completion()
            return
        self._completion_items = [
            {
                "value": item.value,
                "label": item.label,
                "description": item.description,
                "kind": item.kind,
                "source": item.source,
            }
            for item in suggestions.items
        ]
        self._completion_prefix = suggestions.prefix
        self._completion_kind = suggestions.kind
        self._completion_index = 0
        self._render_slash_completion()

    def _render_slash_completion(self) -> None:
        """把补全项写入编辑器内嵌列表（对齐 TS：列表在编辑器底部边框下方）。"""
        items = [
            (
                str(item.get("value", "")).strip(),
                str(item.get("label", str(item.get("value", ""))).strip()),
            )
            for item in self._completion_items
        ]
        self._editor.set_completion(
            items,
            self._completion_index,
            self._autocomplete_max_visible,
        )
        self._resize_editor_for_completion()

    def _resize_editor_for_completion(self) -> None:
        """补全展开时增高编辑器，dock/文档随之下移（对齐 TS editorContainer 自适应）。"""
        editor = self._editor
        extra = editor._completion_line_count()
        self.screen.set_child_basis(editor, 6 + extra)

    def _hide_slash_completion(self) -> None:
        self._completion_items = []
        self._completion_index = 0
        self._completion_prefix = ""
        self._completion_kind = "text"
        self._editor.clear_completion()
        self.screen.set_child_basis(self._editor, 6)

    def _insert_completion(self, value: str) -> None:
        from pi_tui.autocomplete import AutocompleteItem

        item = AutocompleteItem(value=value, kind=self._completion_kind)
        text, cursor = self._autocomplete_provider.apply_completion(
            self._editor.text,
            item,
            self._completion_prefix,
            self._editor_cursor(self._editor),
        )
        self._set_editor_cursor(text, cursor)

    def _editor_cursor(self, editor) -> int:
        lines = list(editor.lines)
        row = min(editor.cursor_row, len(lines) - 1)
        col = editor.cursor_col if row == editor.cursor_row else len(lines[row])
        return sum(len(line) + 1 for line in lines[:row]) + col

    def _set_editor_cursor(self, text: str, cursor: int) -> None:
        lines = text.split("\n")
        remaining = cursor
        row = 0
        col = 0
        for index, line in enumerate(lines):
            if remaining <= len(line):
                row = index
                col = remaining
                break
            remaining -= len(line) + 1
        else:
            row = len(lines) - 1
            col = len(lines[-1])
        self._editor.lines = lines
        self._editor.cursor_row = row
        self._editor.cursor_col = col
        self._editor.selection_anchor = None
        self._editor.refresh()

    def on_pi_editor_completion_navigate_requested(self, message) -> None:
        if not self._completion_items:
            return
        count = len(self._completion_items)
        self._completion_index = (self._completion_index + message.delta) % count
        self._editor.completion_index = self._completion_index
        self._render_slash_completion()

    def on_pi_editor_completion_submit_requested(self, message) -> None:
        if not self._completion_items:
            return
        if not (0 <= self._completion_index < len(self._completion_items)):
            return
        value = str(self._completion_items[self._completion_index].get("value", ""))
        self._insert_completion(value)
        self._hide_slash_completion()

    def on_pi_editor_completion_hide_requested(self, message) -> None:
        self._hide_slash_completion()

    async def _send_prompt(self, text: str) -> None:
        self._set_status(self._working_message)
        try:
            await self._session.prompt(text)
        except Exception as exc:
            self._notify(f"Prompt failed: {exc}")

    async def _exec_slash(self, text: str) -> None:
        expanded = self._session.expand_prompt(text)
        if expanded != text:
            await self._send_prompt(expanded)
            return
        try:
            await self._slash_registry.execute(text, self._slash_context)
        except Exception as exc:
            self._notify(f"Command failed: {exc}")
        self._update_footer()

    async def _exec_bash(self, text: str) -> None:
        """交互 shell 命令：`!cmd` 执行并进入上下文，`!!cmd` 执行但不进上下文。"""
        is_excluded = text.startswith("!!")
        command = text[2:].strip() if is_excluded else text[1:].strip()
        if not command:
            return
        if self._session.is_bash_running:
            self._notify("A bash command is already running. Press Esc to cancel it first.")
            self._editor.text = text
            return
        entry = BashExecutionEntry(
            command,
            exclude_from_context=is_excluded,
            theme_colors=self._chat_theme_colors,
        )
        self._chat.mount(entry)
        self._chat.scroll_end()
        self._set_status("Running bash")
        try:
            result = await self._session.execute_bash(
                command,
                on_chunk=lambda chunk, _progress: entry.append_output(chunk),
                exclude_from_context=is_excluded,
                shell_path=self._shell_path(),
                command_prefix=self._shell_command_prefix(),
            )
            entry.set_complete(
                result.exit_code,
                cancelled=result.cancelled,
                truncated=result.truncated,
                full_output_path=result.full_output_path,
            )
            self._set_status("Idle")
        except Exception as exc:
            entry.set_error(str(exc))
            self._notify(f"Bash command failed: {exc}")
        self._update_footer()

    def _shell_path(self) -> str | None:
        manager = self._settings_manager
        return manager.get_shell_path() if manager is not None else None

    def _shell_command_prefix(self) -> str | None:
        manager = self._settings_manager
        return manager.get_shell_command_prefix() if manager is not None else None

    # ------------------------------------------------------------------
    # 快捷键 actions
    # ------------------------------------------------------------------

    def action_interrupt(self) -> None:
        if self._session.is_bash_running:
            self._session.abort_bash()
            self._set_status("Aborting bash")
            return
        if self._editor.text.lstrip().startswith("!"):
            self._editor.clear()
            return
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
        self.action_exit()

    def on_pi_editor_copy_requested(self, _message) -> None:
        self.action_copy_last_message()

    def on_copy_requested(self, message) -> None:
        """复制请求：列表弹层选中项 / 聊天消息点击。"""
        text = getattr(message, "text", "")
        if text:
            self._copy_to_clipboard(text)
            self._notify("Copied")

    def on_pi_editor_cycle_thinking_requested(self, _message) -> None:
        self.action_cycle_thinking()

    def action_external_editor(self) -> None:
        """ctrl+g：用外部编辑器编辑当前输入。"""
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
        """弹输入框并等待结果（不阻塞事件循环）。"""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.push_screen(
            TextInputDialog(message, placeholder),
            callback=lambda value: future.set_result(value),
        )
        return await future

    def _open_input_selector(self, pending_text: str | None = None) -> None:
        """/input：打开仅含 user 消息的选择器。"""
        nodes = _user_message_nodes(self._session.session_manager)
        if not nodes:
            self._notify("No user messages to edit")
            return
        self.push_screen(
            TreeSelector(nodes, leaf_id=None),
            callback=lambda entry_id: self._on_input_selected(entry_id, pending_text),
        )

    def _on_input_selected(self, entry_id, pending_text: str | None) -> None:
        if entry_id is None:
            return
        if pending_text is None:
            self._run_task(self._await_input_text(entry_id))
        else:
            self._run_task(self._apply_input(entry_id, pending_text))

    async def _await_input_text(self, entry_id: str) -> None:
        """无参数 /input：先选消息，再弹输入框要合并的文本。"""
        text = await self._await_text_input(
            "Text to merge into the selected message:",
            "type here and press Enter",
        )
        if text is None or not text.strip():
            return
        await self._apply_input(entry_id, text.strip())

    async def _apply_input(self, entry_id: str, text: str) -> None:
        """挂起当前任务 → 合并文本到目标消息 → 重建会话 → 继续任务。"""
        try:
            session = self._session
            await session.abort()
            await session.wait_for_idle()
            manager = session.session_manager
            from ..._session_manager_v4 import edit_session_message

            await edit_session_message(manager, entry_id, text)
            new_session = await self._apply_rebuilt_session(manager)
            self._set_status("Continuing from edited message")
            await new_session.continue_()
            self._update_footer()
        except Exception as exc:
            self._notify(f"/input failed: {exc}")

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
            self._render_missed_summaries()
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

    def _open_trust_selector(self) -> None:
        if self._trust_manager is None:
            self._notify("Trust manager not available")
            return
        cwd = self._session.cwd
        # 信任选项由 pi_coding_agent 构造并注入（pi_tui 保持分层独立）。
        from ...trust import get_project_trust_options

        options = get_project_trust_options(cwd)
        self.push_screen(
            TrustSelector(
                cwd,
                options,
                saved_decision=self._trust_manager.get_trust_entry(cwd),
                project_trusted=self._project_trusted,
            ),
            callback=self._on_trust_selected,
        )

    def _on_trust_selected(self, option) -> None:
        if option is None or self._trust_manager is None:
            return
        updates = option.get("updates") or []
        if updates:
            self._trust_manager.set_many(updates)
        self._project_trusted = bool(option.get("trusted"))
        self._needs_trust_decision = False
        if self._settings_manager is not None:
            self._settings_manager.set_project_trusted(self._project_trusted)
            self._settings = self._settings_manager.as_dict()
        label = option.get("label", "Trust")
        self._notify(f"{label}: saved. Project resources load on /reload or restart.")

    def _open_settings_selector(self) -> None:
        items: list[dict[str, Any]] = [
            {"key": "autoCompaction", "label": "Auto-compact", "type": "bool"},
            {
                "key": "defaultProjectTrust",
                "label": "Default project trust",
                "type": "choice",
                "choices": ["ask", "trust", "block"],
            },
            {"key": "trustOverride", "label": "Trust override", "type": "bool"},
            {"key": "defaultProvider", "label": "Default provider", "type": "string"},
            {"key": "defaultModel", "label": "Default model", "type": "string"},
        ]
        current = dict(self._settings)
        current.setdefault("autoCompaction", self._session.auto_compaction_enabled)
        self.push_screen(
            SettingsSelector(items, current, self._on_settings_change),
            callback=lambda _result: self._update_footer(),
        )

    def _on_settings_change(self, key: str, value) -> None:
        """持久化设置到项目 .pi/settings.json 并应用会话侧效果。"""
        if self._settings_manager is not None:
            try:
                self._settings_manager.set_project_setting(key, value)
            except RuntimeError as exc:
                self._notify(str(exc))
                return
            self._settings = self._settings_manager.as_dict()
            if key == "autoCompaction":
                self._session.set_auto_compaction_enabled(bool(value))
            self._notify(f"Saved {key} = {value}")
            return
        from ..._config import _load_json, get_project_settings_path

        cwd = self._session.cwd
        path = get_project_settings_path(cwd)
        data = _load_json(path)
        data[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self._notify(f"Settings save failed: {exc}")
            return
        self._settings[key] = value
        if key == "autoCompaction":
            self._session.set_auto_compaction_enabled(bool(value))
        self._notify(f"Saved {key} = {value} to {path}")

    def _open_thinking_selector(self) -> None:
        levels = self._session.get_available_thinking_levels()
        self.push_screen(
            ThinkingSelector(list(levels), current=self._session.thinking_level),
            callback=self._on_thinking_selected,
        )

    def _on_thinking_selected(self, level) -> None:
        if level is None:
            return
        self._session.set_thinking_level(level)
        self._update_footer()
        self._notify(f"Thinking level: {level}")

    async def _open_oauth_selector(self, mode: str = "login") -> None:
        from pi_ai.auth.oauth import builtin_oauth_providers

        store = self._model_runtime.auth_store
        logged_in: set[str] = set()
        if store is not None:
            try:
                infos = await store.list()
                logged_in = {info["provider_id"] for info in infos}
            except Exception:
                pass
        providers = [
            (provider_id, name, provider_id in logged_in)
            for provider_id, name, _flow in builtin_oauth_providers()
        ]
        self.push_screen(
            OAuthSelector(providers, mode=mode),
            callback=lambda provider_id: self._on_oauth_selected(provider_id, mode),
        )

    def _on_oauth_selected(self, provider_id, mode: str) -> None:
        if provider_id is None:
            return
        self._run_task(
            self._exec_slash(f"/{'logout' if mode == 'logout' else 'login'} {provider_id}")
        )

    async def _open_scoped_models_selector(self) -> None:
        models = await self._model_runtime.get_available()
        if not models:
            self._notify("No models available")
            return
        selected = {
            (scoped.model.provider, scoped.model.id) for scoped in self._session.scoped_models
        }
        self.push_screen(
            ScopedModelsSelector(models, selected, current=self._session.model),
            callback=self._on_scoped_models_selected,
        )

    def _on_scoped_models_selected(self, selected) -> None:
        if selected is None:
            return
        from ...model_resolver import ScopedModel

        if not selected:
            self._session.set_scoped_models([])
            self._notify("Scoped models cleared (all available are usable)")
            return
        by_key = {
            (model.provider, model.id): model
            for model in self._model_runtime.get_available_snapshot()
        }
        scoped = [ScopedModel(model=by_key[key]) for key in selected if key in by_key]
        self._session.set_scoped_models(scoped)
        self._notify(f"Scoped {len(scoped)} models")

    def _open_extensions_selector(self) -> None:
        runner = self._session.extension_runner
        extensions = list(runner.extensions) if runner is not None else []
        if not extensions:
            self._notify("No extensions loaded")
            return
        entries = []
        for extension in extensions:
            from pathlib import Path as _Path

            path = extension.path
            label = _Path(path).name if _Path(path).name else path
            commands = getattr(extension, "commands", None) or {}
            tools = getattr(extension, "tools", None) or {}
            entries.append(
                {
                    "path": path,
                    "label": f"{label} ({len(commands)} commands, {len(tools)} tools)",
                }
            )
        self.push_screen(
            ExtensionSelector(entries),
            callback=self._on_extension_selected,
        )

    def _on_extension_selected(self, entry) -> None:
        if entry is None:
            return
        self._notify(f"Extension: {entry['path']}")
        self._chat.add_message_agent(
            {
                "role": "system",
                "content": f"Extension: {entry['path']}",
            }
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
            from ..._session_manager_v4 import fork_session_manager

            forked = await fork_session_manager(manager, entry_id)
            new_session = await self._apply_rebuilt_session(forked)
            self._slash_context.session = new_session
            self._notify(f"Forked at {entry_id[:8]} (session {new_session.session_id})")
        except Exception as exc:
            self._notify(f"Fork failed: {exc}")

    def action_toggle_tools(self) -> None:
        """切换工具输出展开/折叠（对齐 TS toggleToolOutputExpansion）。"""
        self._tools_expanded = not self._tools_expanded
        self._show_tools = not self._show_tools
        self._header.set_expanded(self._tools_expanded)
        self._chat.set_visibility(
            show_tools=self._show_tools,
            show_thinking=self._show_thinking,
        )
        for widget in self._chat.query(ToolExecutionEntry) + self._chat.query(BashExecutionEntry):
            if isinstance(widget, (ToolExecutionEntry, BashExecutionEntry)):
                widget.set_expanded(self._tools_expanded)
        self._set_status(f"Tool output: {'expanded' if self._tools_expanded else 'collapsed'}")

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        if self._settings_manager is not None:
            self._settings_manager.set_hide_thinking_block(not self._show_thinking)
        self._rerender_chat()
        self._set_status(f"Thinking blocks: {'hidden' if not self._show_thinking else 'visible'}")

    def action_previous_prompt(self) -> None:
        self._scroll_to_prompt(-1)

    def action_next_prompt(self) -> None:
        self._scroll_to_prompt(1)

    def _scroll_to_prompt(self, direction: int) -> None:
        """对齐 TS scrollToPrompt：按 OSC133 语义在 user prompt 间滚动。"""
        body = self._chat._body()
        rows: list[int] = []
        y = 0
        for widget in body.children:
            if isinstance(widget, MessageEntry) and widget.label == "User":
                rows.append(y)
            y += widget.content_size()[1]
        if not rows:
            return
        current = self._chat.scroll_offset
        target: int | None = None
        if direction < 0:
            for row in reversed(rows):
                if row < current:
                    target = row
                    break
        else:
            for row in rows:
                if row > current:
                    target = row
                    break
        if target is None:
            return
        self._chat.scroll_offset = max(0, target)
        self._chat.refresh()

    def _rerender_chat(self) -> None:
        self._chat.set_visibility(show_tools=self._show_tools, show_thinking=self._show_thinking)
        self._chat.clear_messages()
        self._tool_entries.clear()
        for message in self._session.get_messages():
            if message.get("role") == "assistant":
                self._render_tool_calls(message)
            self._chat.add_message_agent(
                cast(dict[str, Any], message),
                skip_tool_calls=(message.get("role") == "assistant"),
            )

    def action_follow_up(self) -> None:
        text = self._editor.text.strip()
        if not text:
            return
        self._editor.clear()
        self._follow_up_queue.append(text)
        self._update_pending_messages()
        self._session.follow_up(text)
        self._set_status("Follow-up queued")

    def action_dequeue(self) -> None:
        self._follow_up_queue.clear()
        self._update_pending_messages()
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
            # 图片处理函数由 pi_coding_agent 注入（pi_tui 不反向依赖 pi_agent）。
            from pi_agent.tools.image_pipeline import process_image_sync

            processed = await asyncio.to_thread(ClipboardImage.process, data, process_image_sync)
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

    async def _dispose_session(self, session: AgentSession) -> None:
        """释放会话：中止运行、等待持久化写入、关闭扩展与 session manager。

        对齐 TS interactive shutdown 的 runtimeHost.dispose() 语义。
        """
        try:
            await session.dispose()
        except Exception:
            pass

    async def _create_new_session(self) -> None:
        try:
            factory = self._session_factory
            if factory is None:
                self._notify("New session not available")
                return
            result = factory()
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            self._notify(f"New session failed: {exc}")
            return
        # 旧会话与新会话使用不同的 session manager：先释放旧会话。
        await self._dispose_session(self._session)
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._autocomplete_provider = self._build_autocomplete_provider()
        self._chat.clear_messages()
        self._update_footer()
        self._set_status("New session")

    async def _apply_rebuilt_session(self, manager):
        """按给定 SessionManager 重建会话并替换。"""
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

        skill_loader = session.skill_loader
        if skill_loader is not None:
            try:
                skill_loader.set_project_dir(
                    Path(session.cwd) / ".pi" / "skills" if self._project_trusted else None
                )
                result = skill_loader.reload()
                details.append(f"{len(result.skills)} skills")
            except Exception as exc:
                details.append(f"skills failed: {exc}")
        template_loader = session.template_loader
        if template_loader is not None:
            try:
                template_loader.set_project_dir(
                    Path(session.cwd) / ".pi" / "prompts" if self._project_trusted else None
                )
                templates = template_loader.reload()
                details.append(f"{len(templates)} prompts")
            except Exception as exc:
                details.append(f"prompts failed: {exc}")

        new_runner = None
        if self._extension_loader is not None:
            old_runner = session.extension_runner
            if old_runner is not None:
                try:
                    await old_runner.shutdown_all()
                except Exception:
                    pass
            try:
                self._extension_loader.set_project_dir(
                    Path(session.cwd) / ".pi" / "extensions" if self._project_trusted else None
                )
                result = await self._extension_loader.load()
                new_runner = ExtensionRunner(
                    result.extensions,
                    runtime=result.runtime,
                    cwd=session.cwd,
                    model_runtime=self._model_runtime,
                )
                session.set_extension_runner(new_runner)
                from .ui_context import TuiUIContext

                new_runner.bind(ui_context=TuiUIContext(self))
                if session.extension_state is not None:
                    session.extension_state["runner"] = new_runner
                await new_runner.discover_resources()
                self._chat.set_renderers(
                    custom_renderer=self._extension_custom_message_renderer,
                    markdown_transformers=self._extension_markdown_transformers,
                    tool_renderer=self._extension_tool_renderer,
                )
                details.append(f"{len(result.extensions)} extensions")
                for error in result.errors:
                    message = error.error.replace("\n", " ")
                    details.append(f"extension error: {message} ({error.extension_path})")
            except Exception as exc:
                details.append(f"extensions failed: {exc}")
        elif session.extension_runner is not None:
            new_runner = session.extension_runner

        # 快捷键 + 扩展命令：重建注册表与键位表。
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
            self._autocomplete_provider = self._build_autocomplete_provider()
            if self._header is not None:
                self._header.refresh_hints()
            details.append("keybindings refreshed")
        except Exception as exc:
            details.append(f"keybindings failed: {exc}")

        # 主题：重新解析并刷新样式。
        try:
            self._theme = self._theme_loader.resolve(self._theme_name)
            self._apply_theme()
            details.append(f"theme {self._theme.name}")
        except Exception as exc:
            details.append(f"theme failed: {exc}")

        try:
            if session.rebuild_system_prompt() is not None:
                details.append("context files + system prompt")
            else:
                details.append("system prompt (static)")
        except Exception as exc:
            details.append(f"system prompt failed: {exc}")

        return "Reloaded: " + "; ".join(details)

    async def _replace_session(self, new_session: AgentSession) -> None:
        old = self._session
        if old is not None and old.session_manager is not new_session.session_manager:
            # 新旧会话共享 session manager 时不 dispose（如 /input 重建场景），
            # 否则释放旧会话：中止运行、等待写入、关闭扩展与 manager。
            await self._dispose_session(old)
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._autocomplete_provider = self._build_autocomplete_provider()
        self._chat.clear_messages()
        self._tool_entries.clear()
        self._rendered_summary_ids = set()
        for message in new_session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))
        self._render_missed_summaries()
        self._update_footer()
        self._set_status(f"Session {new_session.session_id}")

    def action_resume_session(self) -> None:
        self._run_task(self._resume_session())

    async def _resume_session(self) -> None:
        from ..._session_manager_v4 import list_sessions as list_sessions_async

        sessions_dir = get_sessions_dir()
        current_sessions = await list_sessions_async(
            sessions_dir,
            cwd=self._session.cwd,
        )
        all_sessions = await list_sessions_async(sessions_dir)
        if not current_sessions and not all_sessions:
            self._notify("No saved sessions")
            return
        session_path = self._session.session_manager.session_path
        # SessionPickerModel 由 pi_coding_agent 构造并注入（pi_tui 保持分层独立）。
        from .session_selector import SessionPickerModel

        model = SessionPickerModel(
            current_sessions=current_sessions,
            all_sessions=all_sessions,
            current_cwd=self._session.cwd,
            current_session_path=str(session_path) if session_path is not None else None,
        )
        self.push_screen(
            SessionPicker(
                model,
                on_rename=self._rename_session_from_picker,
                on_delete=self._delete_session_from_picker,
                reload_sessions=self._reload_picker_sessions,
                keybindings_manager=self._keybindings,
            ),
            callback=self._on_session_selected,
        )

    async def _rename_session_from_picker(self, path: str, name: str) -> str | None:
        from ..._session_manager_v4 import open_session_manager

        manager = await open_session_manager(path)
        try:
            await manager.append_session_info(name)
        finally:
            close = getattr(manager, "close", None)
            if close is not None:
                result = close()
                if result is not None and inspect.isawaitable(result):
                    await result
        return None

    async def _delete_session_from_picker(self, path: str) -> str | None:
        from .session_selector import delete_session_file

        result = await delete_session_file(path)
        return None if result.ok else result.error

    async def _reload_picker_sessions(self) -> tuple[list, list]:
        from ..._session_manager_v4 import list_sessions as list_sessions_async

        sessions_dir = get_sessions_dir()
        current = await list_sessions_async(
            sessions_dir,
            cwd=self._session.cwd,
        )
        all_sessions = await list_sessions_async(sessions_dir)
        return current, all_sessions

    def _on_session_selected(self, path) -> None:
        if path is None or self._resume_factory is None:
            return
        self._run_task(self._apply_resumed_session(path))

    async def _apply_resumed_session(self, path: str) -> None:
        try:
            factory = self._resume_factory
            if factory is None:
                return
            result = factory(path)
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Resume failed: {exc}")
            return
        # 恢复的会话使用新的 session manager：先释放旧会话。
        await self._dispose_session(self._session)
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._autocomplete_provider = self._build_autocomplete_provider()
        self._chat.clear_messages()
        self._tool_entries.clear()
        self._rendered_summary_ids = set()
        for message in new_session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))
        self._render_missed_summaries()
        self._update_footer()
        self._set_status(f"Resumed {new_session.session_id}")

    def _render_missed_summaries(self) -> None:
        """把会话树里的 branch_summary 条目渲染为聊天消息（不重复）。"""
        try:
            for entry in self._session.session_manager.get_entries():
                if entry.get("type") != "branch_summary":
                    continue
                entry_id = entry.get("id")
                if entry_id in self._rendered_summary_ids:
                    continue
                if not entry_id:
                    continue
                self._rendered_summary_ids.add(entry_id)
                self._chat.add_message_agent(
                    cast(
                        dict[str, Any],
                        {
                            "role": "branchSummary",
                            "summary": str(entry.get("summary", "")),
                        },
                    )
                )
        except Exception:
            pass

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

    async def _handle_event(self, event: KeyEvent) -> None:
        was_resize = event.type == "resize"
        await super()._handle_event(event)
        if was_resize:
            for key in list(self._overlay_renderers):
                self._render_overlay_renderer(key)


def _list_sessions() -> list[dict[str, Any]]:
    """按修改时间倒序列出会话文件（per-cwd 布局，自动迁移旧平铺文件）。"""
    infos = SessionManager.list_sessions(get_sessions_dir())
    return [
        {
            "path": info.path,
            "session_id": info.session_id,
            "cwd": info.cwd,
            "modified": info.modified,
        }
        for info in infos
    ]


def _user_message_nodes(manager: SessionManagerLike) -> list[SessionTreeNode]:
    """构造仅含 user 消息的扁平树节点（/input 选择器用）。"""
    nodes: list[SessionTreeNode] = []
    for entry in manager.get_entries():
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        snippet = " ".join(text.split())
        if len(snippet) > 60:
            snippet = f"{snippet[:60]}…"
        nodes.append(
            SessionTreeNode(
                id=cast(str, entry.get("id")),
                parent_id=entry.get("parentId"),
                entry=entry,
                label=snippet or "(empty)",
            )
        )
    nodes.reverse()  # 最新的 user 消息排在最前。
    return nodes


def _format_context_path(path: str, cwd: str) -> str:
    """启动提示用路径：cwd 内显示相对路径，否则 home 缩写。"""
    try:
        relative = Path(path).resolve().relative_to(Path(cwd).resolve())
        return relative.as_posix()
    except ValueError:
        return shorten_path(path).replace("\\", "/")


def _copy_text(text: str) -> None:
    """尽力而为的剪贴板文本写入（失败静默）。"""
    try:
        if sys.platform == "win32":
            import subprocess

            subprocess.run(["clip"], input=text.encode("utf-16-le") + b"\x00\x00", check=False)
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
    try:
        await app.run_async()
    finally:
        from pi_tui.terminal import drain_pending_osc_response

        drain_pending_osc_response()
        # 对齐 TS interactive shutdown：退出时 dispose 当前会话（app 运行期间
        # 可能已通过 new/resume 切换），等待 pending writes、关闭扩展与 manager。
        current = getattr(app, "_session", None)
        if current is not None:
            try:
                await current.dispose()
            except Exception:
                pass
    return 0


__all__ = ["PiTuiApp", "run_tui_mode", "_list_sessions"]
