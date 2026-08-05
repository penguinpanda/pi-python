"""pi TUI 主应用（对齐 TS modes/interactive/）。"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

from textual.app import App, AwaitComplete, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.geometry import Offset
from textual.widgets import Static

from ..._config import get_sessions_dir
from ..._session import AgentSession
from ..._session_manager import SessionManager, SessionTreeNode
from pi_agent import AgentEvent
from ...model_runtime import ModelRuntime
from ...extensions import ExtensionRunner
from ...extensions.registry import ExtensionRegistry
from pi_tui.clipboard_image import ClipboardImage
from pi_tui.autocomplete import CombinedAutocompleteProvider
from pi_tui.components import (
    BashExecutionEntry,
    MessageEntry,
    PiChatContainer,
    PiEditor,
    PiFooter,
    PiHeader,
    PiStatusBar,
)
from pi_tui.keybindings import KeybindingsManager
from pi_tui.overlay import (
    OverlayHandle,
    OverlayHooks,
    OverlayManager,
    OverlayRect,
    OverlayWidget,
)
from pi_tui.selectors import (
    ChoiceSelector,
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

OverlayDialog {
    background: __PI_BGPANEL__;
    color: __PI_TEXT__;
    border: round __PI_BORDERACTIVE__;
    padding: 0 1;
    height: auto;
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
        settings_manager=None,
        trust_manager=None,
        project_trusted: bool = False,
        needs_trust_decision: bool = False,
    ) -> None:
        self._keybindings = keybindings_manager or KeybindingsManager()
        self._settings = settings if settings is not None else {}
        self._settings_manager = settings_manager
        if self._settings:
            self._keybindings.load_from_settings(self._settings)
        self._theme_loader = theme_loader or ThemeLoader()
        self._theme_name = theme_name
        self._theme = self._theme_loader.resolve(theme_name)
        self._extension_loader = extension_loader
        self._trust_manager = trust_manager
        self._project_trusted = project_trusted
        self._needs_trust_decision = needs_trust_decision

        # 实例级 BINDINGS / CSS：必须在 super().__init__() 之前设置。
        self.BINDINGS = [  # type: ignore[misc]
            Binding(binding.key, binding.action, binding.description)
            for binding in self._keybindings.all_bindings()
        ]
        self.CSS = _build_css(self._theme.colors)  # type: ignore[misc]
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
        self._rendered_summary_ids: set[str] = set()
        self._custom_editor: PiEditor | None = None
        self._widget_above: dict[str, str] = {}
        self._widget_below: dict[str, str] = {}
        self._overlay_dialog_callbacks: dict[str, Callable[[Any], None] | None] = {}
        self._overlay_renderers: dict[str, Callable[[int, int], list[str]]] = {}
        self._overlay_manager = OverlayManager(
            OverlayHooks(
                make_widget=lambda key, lines, options: OverlayWidget(key, lines, options),
                update_widget=self._update_overlay_widget,
                make_component_widget=lambda key, component, options: OverlayWidget(
                    key, [], options, component=component
                ),
                update_component=self._update_overlay_component_widget,
                mount=self._mount_overlay,
                remove=lambda widget: widget.remove(),
                set_visible=lambda widget, visible: setattr(widget, "display", visible),
                reposition=self._apply_overlay_rect,
                focus=lambda widget: widget.focus(),
                current_focus=lambda: self.screen.focused,
                content_size=lambda widget: (
                    widget.content_size.width,
                    widget.content_size.height,
                ),
                bring_to_front=self._bring_overlay_to_front,
            )
        )
        self._hidden_thinking_label = "Thinking"
        self._working_message = "Working"
        self._stream_entry: MessageEntry | None = None

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

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield PiHeader(self._keybindings, id="pi-header")
        yield PiChatContainer(id="pi-chat")
        yield Static("", id="pi-widgets-above")
        yield PiStatusBar("Idle", id="pi-status")
        yield PiEditor(id="pi-editor")
        yield Static("", id="pi-widgets-below")
        yield PiFooter("", id="pi-footer")

    def on_mount(self) -> None:
        self._bind_session()
        for message in self._session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))
        self._render_missed_summaries()
        self._update_footer()
        self._editor.focus()
        # 启动时对未定信任项目提示（对齐 TS 启动 trust 选择器）。
        if self._needs_trust_decision:
            self.call_after_refresh(self._open_trust_selector)

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
        self._chat.set_renderers(
            custom_renderer=self._extension_custom_message_renderer,
            markdown_transformers=self._extension_markdown_transformers,
            tool_renderer=self._extension_tool_renderer,
        )
        self._unsubscribe = self._session.subscribe(
            cast(Callable[[AgentEvent], None], self._on_session_event)
        )

    # ------------------------------------------------------------------
    # 组件快捷访问
    # ------------------------------------------------------------------

    @property
    def _chat(self) -> PiChatContainer:
        return self.query_one("#pi-chat", PiChatContainer)

    @property
    def _editor(self) -> PiEditor:
        if self._custom_editor is not None:
            return self._custom_editor
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
            if event_type == "message_start":
                self._begin_stream()
            elif event_type == "message_update":
                self._update_stream(event.get("message"))
            elif event_type == "message_end":
                self._finish_stream()
                message = event.get("message")
                if message is not None:
                    self._chat.add_message_agent(message)
                self._update_footer()
            elif event_type == "agent_settled":
                self._finish_stream()
                self._set_status("Idle")
                self._update_footer()
            elif event_type == "compaction_start":
                self._set_status("Compacting")
            elif event_type in ("compaction_end",):
                self._set_status("Idle")
            elif event_type in ("model_changed", "thinking_level_changed"):
                self._update_footer()
            elif event_type == "agent_start":
                self._set_status(self._working_message)
            elif event_type == "skill_invocation":
                skill = event.get("skill", "")
                self._chat.add_message_agent(
                    {
                        "role": "skillInvocation",
                        "content": f"Invoked skill: {skill}",
                    }
                )
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
        self._chat.scroll_end(animate=False)

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
            elif block_type == "thinking" and block.get("thinking"):
                parts.append(str(block.get("thinking", "")))
            elif block_type == "toolCall":
                parts.append(f"{block.get('name', 'tool')}({block.get('arguments', {})})")
        self._stream_entry.set_text("\n\n".join(parts))
        self._chat.scroll_end(animate=False)

    def _finish_stream(self) -> None:
        """消息结束：移除流式占位（最终消息由 message_end 正常追加）。"""
        entry = self._stream_entry
        self._stream_entry = None
        if entry is not None and entry.is_mounted:
            try:
                entry.remove()
            except Exception:
                pass

    def _extension_custom_message_renderer(self, message):
        """custom 角色消息 → 扩展注册的消息渲染器（返回文本或 None）。"""
        runner = self._session.extension_runner
        if runner is None:
            return None
        renderer = runner.get_message_renderer(str(message.get("customType", "")))
        if renderer is None:
            return None
        result = renderer(message)
        if asyncio.iscoroutine(result):
            # 渲染是同步路径；异步渲染器暂不等待。
            return None
        return result

    def _extension_markdown_transformers(self):
        """扩展注册的 markdown 变换器链。"""
        runner = self._session.extension_runner
        if runner is None:
            return []
        return runner.get_markdown_transformers()

    def _extension_tool_renderer(self, message):
        """内置工具结果 → 扩展注册的工具渲染器（返回字符串或 None）。"""
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
        """用扩展提供的编辑器组件替换 PiEditor（对齐 TS setEditorComponent）。

        组件必须是 PiEditor 子类（继承提交/快捷键语义），替换后聚焦新编辑器。
        """
        from pi_tui import PiEditor as PiEditorType

        if not isinstance(component, PiEditorType):
            raise TypeError("set_editor_component requires a PiEditor subclass")
        old = self.query_one("#pi-editor", PiEditorType)
        old.display = False
        component.id = f"pi-editor-{id(component):x}"
        footer = self.query_one("#pi-footer")
        self.screen.mount(component, before=footer)
        self._custom_editor = component
        component.focus()

    def _set_status(self, text: str) -> None:
        self._status.update(text)

    def _set_widget(self, key: str, lines: list[str], options: dict | None = None) -> None:
        """编辑器上方（默认）或下方显示多行组件（对齐 TS setWidget）。"""
        below = (options or {}).get("placement") == "belowEditor"
        target = self._widget_below if below else self._widget_above
        if lines:
            target[key] = "\n".join(lines)
        else:
            target.pop(key, None)
        widget = self.query_one(
            "#pi-widgets-below" if below else "#pi-widgets-above",
            Static,
        )
        widget.update("\n".join(target.values()))

    def _set_hidden_thinking_label(self, label: str | None = None) -> None:
        """设置折叠 thinking 块的标签（None 恢复默认 "Thinking"）。"""
        self._hidden_thinking_label = label or "Thinking"
        self._chat.set_hidden_thinking_label(self._hidden_thinking_label)

    def _set_working_message(self, text: str | None = None) -> None:
        """设置流式工作提示文案（None 恢复默认 "Working"）。"""
        self._working_message = text or "Working"

    def _set_theme(self, theme: str | None = None) -> None:
        """切换主题（None 恢复当前配置主题）。"""
        try:
            self._theme = self._theme_loader.resolve(theme or self._theme_name)
            self.CSS = _build_css(self._theme.colors)  # type: ignore[misc]
            self.refresh_css()
        except Exception:
            pass

    def _set_overlay(
        self,
        key: str,
        lines: list[str],
        options: dict | None = None,
    ) -> OverlayHandle | None:
        """显示 / 更新浮层（OverlayManager + overlay 层）；空列表移除。"""
        if not lines:
            self._overlay_renderers.pop(key, None)
            self._overlay_manager.remove(key)
            return None
        self._overlay_renderers.pop(key, None)
        handle = self._overlay_manager.show(key, list(lines), options or {})
        self.call_after_refresh(lambda: self._overlay_manager.reposition(key))
        return handle

    def _set_overlay_component(
        self,
        key: str,
        component,
        options: dict | None = None,
    ) -> OverlayHandle | None:
        """用任意 Textual 组件作为 overlay（组件树 API）；None 移除。"""
        if component is None:
            self._overlay_renderers.pop(key, None)
            self._overlay_manager.remove(key)
            return None
        self._overlay_renderers.pop(key, None)
        handle = self._overlay_manager.show_component(key, component, options or {})
        self.call_after_refresh(lambda: self._overlay_manager.reposition(key))
        self.call_after_refresh(lambda: self._overlay_manager.ensure_focus(key))
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
        self.call_after_refresh(lambda: self._render_overlay_renderer(key))
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

    def _update_overlay_widget(
        self,
        widget: OverlayWidget,
        lines: list[str],
        options,
    ) -> None:
        widget.update_options(options)
        widget.update_content(lines)

    def _update_overlay_component_widget(self, widget, component, options) -> None:
        widget.update_options(options)
        widget.set_component(component)

    def _mount_overlay(self, widget: OverlayWidget) -> None:
        self.screen.mount(widget)

    def _apply_overlay_rect(
        self,
        widget: OverlayWidget,
        rect: OverlayRect,
        options,
    ) -> None:
        """把解析后的绝对矩形应用到 overlay widget（含动画）。"""
        target = Offset(rect.col, rect.row)
        if options.behavior.animate:
            widget.animate(
                "offset",
                target,
                duration=float(options.behavior.duration),
                easing="out_cubic",
            )
        else:
            widget.styles.offset = target

    def _bring_overlay_to_front(self, widget: OverlayWidget) -> None:
        """重新挂载到屏幕末尾，确保 focusOrder 置顶（DOM 顺序决定同层堆叠）。"""

        async def _reorder() -> None:
            try:
                if widget.is_attached:
                    await widget.remove()
                await self.screen.mount(widget)
            except Exception:
                pass
            finally:
                # 重挂载会重建组件子树，焦点可能被 Textual 挪走；完成后重新落位。
                entry = self._overlay_manager.entry_for_widget(widget)
                if entry is not None:
                    self._overlay_manager.ensure_focus(entry.key)

        self._run_task(_reorder())

    def on_descendant_focus(self, event) -> None:
        """Textual 焦点变化 → overlay 焦点状态机同步。"""
        self._overlay_manager.on_widget_focused(event.widget)

    def on_resize(self, event) -> None:
        """终端尺寸变化 → overlay 重排 + 可见性 / 焦点重定向。"""
        size = event.size
        self._overlay_manager.on_resize((size.width, size.height))
        for key in list(self._overlay_renderers):
            self.call_after_refresh(lambda k=key: self._render_overlay_renderer(k))

    def on_key(self, event) -> None:
        """输入前焦点恢复（blocked/active 状态下回到 overlay）。"""
        self._overlay_manager.route_input()

    def push_screen(self, screen, callback=None, wait_for_dismiss=False, *, mode=None) -> Any:
        """选择器改走 overlay 层；其余（内置帮助等）仍走屏幕栈。"""
        if isinstance(
            screen,
            (
                ChoiceSelector,
                TextInputDialog,
                ThinkingSelector,
                SettingsSelector,
                ModelSelector,
                SessionPicker,
                TreeSelector,
                OAuthSelector,
                ScopedModelsSelector,
                ExtensionSelector,
                TrustSelector,
            ),
        ):
            self._open_overlay_selector(screen, callback)
            return None
        return super().push_screen(
            screen,
            callback=callback,
            wait_for_dismiss=wait_for_dismiss,
            mode=mode,
        )

    def _open_overlay_selector(self, component, callback=None) -> None:
        """把对话框组件挂进 overlay 层（选择器即 overlay）。"""
        key = f"dialog-{id(component):x}"
        self._overlay_dialog_callbacks[key] = callback
        self._overlay_manager.show_component(
            key,
            component,
            {"anchor": "center", "width": "80%", "maxHeight": "60%"},
        )
        self.call_after_refresh(lambda: self._overlay_manager.reposition(key))
        self.call_after_refresh(lambda: self._overlay_manager.ensure_focus(key))

    def _close_overlay_dialog(self, component, value=None) -> None:
        """对话框 dismiss：移除 overlay 并回调结果。"""
        entry = self._overlay_manager.entry_for_widget(component)
        if entry is None:
            return
        key = entry.key
        callback = self._overlay_dialog_callbacks.pop(key, None)
        self._overlay_manager.remove(key)
        if callback is not None:
            callback(value)

    def pop_screen(self) -> AwaitComplete:
        """弹出 ModalScreen 后立即同步 overlay 焦点（Textual 原生恢复 + route_input）。"""
        result = super().pop_screen()
        self.call_after_refresh(self._overlay_manager.route_input)
        return result

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
                    self._chat.scroll_to_widget(entry, animate=False)
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
        elif text.startswith("!"):
            self._run_task(self._exec_bash(text))
        else:
            self._run_task(self._send_prompt(text))

    async def on_pi_editor_autocomplete_requested(
        self,
        message: PiEditor.AutocompleteRequested,
    ) -> None:
        """Tab：查询扩展自动补全 provider，弹选择器并插入选中值。"""
        runner = self._session.extension_runner
        if runner is None:
            return
        providers = runner.get_autocomplete()
        if not providers:
            return
        completions = await CombinedAutocompleteProvider(providers).collect(message.editor.text)
        if not completions:
            return

        def _label(item: dict) -> str:
            return str(item.get("label", item.get("value", "")))

        def _callback(selected) -> None:
            if selected is None:
                return
            match = next(
                (item for item in completions if _label(item) == selected),
                None,
            )
            if match is None:
                return
            value = str(match.get("value", match.get("label", "")))
            try:
                self._editor.insert(value)
            except Exception:
                self._editor.text += value

        self.push_screen(
            ChoiceSelector("Autocomplete", [_label(item) for item in completions]),
            callback=_callback,
        )

    async def _send_prompt(self, text: str) -> None:
        self._set_status(self._working_message)
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
        entry = BashExecutionEntry(command, exclude_from_context=is_excluded)
        self._chat.mount(entry)
        self._chat.scroll_end(animate=False)
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
            # 1. 挂起：运行中则中止并等待本轮结束（空闲时是 no-op）。
            await session.abort()
            await session.wait_for_idle()
            # 2. 合并文本并回卷 leaf 到目标消息。
            manager = session.session_manager
            manager.edit_message(entry_id, text)
            # 3. 重建会话（从编辑后的消息重新加载上下文）并继续。
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
        self.push_screen(
            TrustSelector(
                cwd,
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
            {
                "key": "autoCompaction",
                "label": "Auto-compact",
                "type": "bool",
            },
            {
                "key": "defaultProjectTrust",
                "label": "Default project trust",
                "type": "choice",
                "choices": ["ask", "trust", "block"],
            },
            {
                "key": "trustOverride",
                "label": "Trust override",
                "type": "bool",
            },
            {
                "key": "defaultProvider",
                "label": "Default provider",
                "type": "string",
            },
            {
                "key": "defaultModel",
                "label": "Default model",
                "type": "string",
            },
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
        from ..._config import (
            _load_json,
            get_project_settings_path,
        )

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
        self._chat.set_visibility(show_tools=self._show_tools, show_thinking=self._show_thinking)
        self._chat.clear_messages()
        for message in self._session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))

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
            factory = self._session_factory
            if factory is None:
                self._notify("New session not available")
                return
            result = factory()
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
            self.BINDINGS = [  # type: ignore[misc]
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
            self.CSS = _build_css(self._theme.colors)  # type: ignore[misc]
            self.refresh_css()
            details.append(f"theme {self._theme.name}")
        except Exception as exc:
            details.append(f"theme failed: {exc}")

        # 5. 系统提示 / 上下文文件：随技能与 AGENTS.md 变化重建。
        try:
            if session.rebuild_system_prompt() is not None:
                details.append("context files + system prompt")
            else:
                details.append("system prompt (static)")
        except Exception as exc:
            details.append(f"system prompt failed: {exc}")

        return "Reloaded: " + "; ".join(details)

    async def _replace_session(self, new_session: AgentSession) -> None:
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._chat.clear_messages()
        self._rendered_summary_ids = set()
        for message in new_session.get_messages():
            self._chat.add_message_agent(cast(dict[str, Any], message))
        self._render_missed_summaries()
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
            factory = self._resume_factory
            if factory is None:
                return
            result = factory(path)
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            self._notify(f"Resume failed: {exc}")
            return
        self._session = new_session
        self._slash_context.session = new_session
        self._bind_session()
        self._chat.clear_messages()
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

    def _run_task(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


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


def _user_message_nodes(manager: SessionManager) -> list[SessionTreeNode]:
    """构造仅含 user 消息的扁平树节点（/input 选择器用，label 显示内容片段）。"""
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
        # 转义方括号，避免被 Textual 当成 Rich markup 样式标签吞掉。
        snippet = snippet.replace("[", r"\[").replace("]", r"\]")
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
    await app.run_async()
    return 0


__all__ = ["PiTuiApp", "run_tui_mode", "_list_sessions"]
