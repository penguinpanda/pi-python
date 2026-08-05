"""TUI 组件：Header / ChatContainer / Editor / StatusBar / Footer。"""

from __future__ import annotations

from typing import Any

from textual import events
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Input, Label, Static, TextArea

from .keybindings import KeybindingsManager
from .markdown import render_labeled_markdown


class CopyRequested(Message):
    """聊天消息请求复制完整文本（点击消息触发，事件冒泡到宿主）。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


# ---------------------------------------------------------------------------
# 消息渲染
# ---------------------------------------------------------------------------


def _block_text(block: dict[str, Any]) -> str:
    """单个内容块 → 文本。"""
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "thinking":
        return block.get("thinking", "")
    if block_type == "toolCall":
        return f"{block.get('name', 'tool')}({block.get('arguments', {})})"
    if block_type == "image":
        return "[image]"
    return ""


def _apply_markdown_transformers(
    text: str,
    transformers,
    message_type: str,
    is_streaming: bool = False,
) -> str:
    """按扩展注册顺序应用 markdown 变换器（单个失败保留上一步结果）。"""
    if not transformers:
        return text
    resolved = transformers() if callable(transformers) else list(transformers or [])
    current = text
    for transformer in resolved:
        try:
            result = transformer(
                current, {"messageType": message_type, "isStreaming": is_streaming}
            )
        except Exception:
            continue
        if isinstance(result, str):
            current = result
    return current


def message_to_entries(
    message: dict[str, Any],
    *,
    show_tools: bool = True,
    show_thinking: bool = True,
    custom_renderer=None,
    markdown_transformers=None,
    hidden_thinking_label: str = "Thinking",
    tool_renderer=None,
) -> list[tuple[str, str]]:
    """AgentMessage → [(label, text)]，供聊天容器渲染。"""
    role = message.get("role")
    content = message.get("content")

    if role == "user":
        if isinstance(content, str):
            return [("User", _apply_markdown_transformers(content, markdown_transformers, "user"))]
        text = "\n".join(_block_text(block) for block in content or [])
        return [("User", _apply_markdown_transformers(text, markdown_transformers, "user"))]

    if role == "toolResult":
        tool_name = message.get("tool_name", "tool")
        label = f"Tool: {tool_name}"
        if message.get("is_error"):
            label += " (error)"
        if tool_renderer is not None:
            rendered = tool_renderer(message)
            if isinstance(rendered, str) and rendered:
                return [(label, rendered)]
        if isinstance(content, str):
            return [(label, content)]
        text = "\n".join(_block_text(block) for block in content or [])
        return [(label, text)]

    if role == "assistant":
        entries: list[tuple[str, str]] = []
        label = "Assistant (error)" if message.get("error_message") else "Assistant"
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[str] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                if show_thinking:
                    thinking_parts.append(block.get("thinking", ""))
            elif block_type == "toolCall":
                if show_tools:
                    name = block.get("name", "tool")
                    arguments = block.get("arguments") or {}
                    tool_calls.append(f"{name}({arguments})")
            elif block_type == "text":
                text_parts.append(block.get("text", ""))
        if thinking_parts:
            entries.append(
                (
                    hidden_thinking_label,
                    _apply_markdown_transformers(
                        "\n".join(thinking_parts), markdown_transformers, "assistant-thinking"
                    ),
                )
            )
        if text_parts:
            entries.append(
                (
                    label,
                    _apply_markdown_transformers(
                        "\n".join(text_parts), markdown_transformers, "assistant"
                    ),
                )
            )
        for tool_call in tool_calls:
            entries.append(("Tool call", tool_call))
        return entries

    if role == "compactionSummary":
        return [("Compaction summary", message.get("summary", ""))]
    if role == "branchSummary":
        return [("Branch summary", message.get("summary", ""))]
    if role == "skillInvocation":
        return [("Skill", message.get("content", ""))]
    if role == "bashExecution":
        command = str(message.get("command", ""))
        output = str(message.get("output", ""))
        status_parts: list[str] = []
        if message.get("cancelled"):
            status_parts.append("(cancelled)")
        elif message.get("exitCode") not in (None, 0):
            status_parts.append(f"(exit {message.get('exitCode')})")
        if message.get("truncated") and message.get("fullOutputPath"):
            status_parts.append(f"Output truncated. Full output: {message.get('fullOutputPath')}")
        lines = [f"$ {command}", output or "(no output)"]
        if status_parts:
            lines.append(" ".join(status_parts))
        label = "Bash (excluded)" if message.get("excludeFromContext") else "Bash"
        return [(label, "\n".join(lines))]
    if role == "system":
        return [("System", message.get("content", ""))]
    if role == "custom":
        custom_type = str(message.get("customType", "custom"))
        if custom_renderer is not None:
            rendered = custom_renderer(message)
            if isinstance(rendered, str) and rendered:
                return [(custom_type, rendered)]
        if isinstance(content, str):
            return [(custom_type, content)]
        text = "\n".join(_block_text(block) for block in content or [])
        return [(custom_type, text)] if text else []
    # 其它角色（custom/branchSummary 等）降级为文本。
    if isinstance(content, str):
        return [("Agent", content)]
    text = "\n".join(_block_text(block) for block in content or [])
    return [("Agent", text)] if text else []


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------


class MessageEntry(Static):
    """聊天消息条目。"""

    def __init__(self, label: str, text: str, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.label = label
        self.entry_text = text
        self.speaking = False

    def on_mount(self) -> None:
        self._refresh_display()

    def set_text(self, text: str) -> None:
        """流式更新正文（未挂载时只记录，挂载后重渲染）。"""
        self.entry_text = text
        if self.is_mounted:
            self._refresh_display()

    def set_speaking(self, speaking: bool) -> None:
        """标记当前是否正在说话（流式回复中显示 Speaking…）。"""
        self.speaking = speaking
        if self.is_mounted:
            self._refresh_display()

    def _refresh_display(self) -> None:
        self.update(render_labeled_markdown(self.label, self.entry_text, speaking=self.speaking))

    async def _on_click(self, event: events.Click) -> None:
        """点击消息 → 复制整条文本（Textual 不支持鼠标选词）。"""
        if self.entry_text:
            self.post_message(CopyRequested(self.entry_text))


class BashExecutionEntry(Static):
    """交互 bash 命令条目：流式输出 + 完成状态。"""

    def __init__(
        self,
        command: str,
        *,
        exclude_from_context: bool = False,
        **kwargs,
    ) -> None:
        super().__init__("", **kwargs)
        self.command = command
        self.exclude_from_context = exclude_from_context
        self.output = ""
        self.status: str | None = None

    @property
    def label(self) -> str:
        return "Bash (excluded)" if self.exclude_from_context else "Bash"

    def on_mount(self) -> None:
        self._update_display()

    def append_output(self, chunk: str) -> None:
        self.output += chunk
        self._schedule_update()

    def set_complete(
        self,
        exit_code: int | None,
        cancelled: bool = False,
        truncated: bool = False,
        full_output_path: str | None = None,
    ) -> None:
        if cancelled:
            self.status = "(cancelled)"
        elif exit_code not in (None, 0):
            self.status = f"(exit {exit_code})"
        else:
            self.status = None
        if truncated and full_output_path:
            prefix = f"{self.status} " if self.status else ""
            self.status = f"{prefix}Output truncated. Full output: {full_output_path}"
        self._schedule_update()

    def set_error(self, message: str) -> None:
        self.status = f"(failed: {message})"
        self._schedule_update()

    def _schedule_update(self) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._update_display)

    def _update_display(self) -> None:
        escaped = self.output.replace("[", r"\[")
        text = f"$ {self.command}"
        if escaped:
            text += f"\n{escaped}"
        if self.status:
            text += f"\n{self.status}"
        self.update(f"[b]{self.label}[/b] {text}")

    async def _on_click(self, event: events.Click) -> None:
        """点击 bash 条目 → 复制命令 + 输出。"""
        parts = [f"$ {self.command}"]
        if self.output:
            parts.append(self.output)
        if self.status:
            parts.append(self.status)
        self.post_message(CopyRequested("\n".join(parts)))


class PiHeader(Static):
    """Logo + 快捷键提示。"""

    def __init__(self, keybindings: KeybindingsManager, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._keybindings = keybindings

    def refresh_hints(self) -> None:
        model_forward = self._keybindings.get_action_key("app.model.cycleForward")
        model_select = self._keybindings.get_action_key("app.model.select")
        exit_key = self._keybindings.get_action_key("app.exit")
        self.update(
            "[b]pi[/b]  "
            f"cycle model {model_forward or ''}  "
            f"select model {model_select or ''}  "
            f"exit {exit_key or ''}"
        )

    def on_mount(self) -> None:
        self.refresh_hints()


class PiChatContainer(VerticalScroll):
    """消息列表容器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._show_tools = True
        self._show_thinking = True
        self._custom_renderer = None
        self._markdown_transformers = None
        self._hidden_thinking_label = "Thinking"
        self._tool_renderer = None

    def set_visibility(self, *, show_tools: bool, show_thinking: bool) -> None:
        self._show_tools = show_tools
        self._show_thinking = show_thinking

    def set_renderers(
        self,
        *,
        custom_renderer=None,
        markdown_transformers=None,
        tool_renderer=None,
    ) -> None:
        """设置扩展渲染器（custom 消息 / markdown 变换器 / 工具结果）。"""
        self._custom_renderer = custom_renderer
        self._markdown_transformers = markdown_transformers
        self._tool_renderer = tool_renderer

    def set_hidden_thinking_label(self, label: str) -> None:
        self._hidden_thinking_label = label

    def set_tool_renderer(self, tool_renderer) -> None:
        self._tool_renderer = tool_renderer

    def add_message_agent(
        self,
        message: dict[str, Any],
        *,
        show_tools: bool | None = None,
        show_thinking: bool | None = None,
    ) -> None:
        entries = message_to_entries(
            message,
            show_tools=self._show_tools if show_tools is None else show_tools,
            show_thinking=self._show_thinking if show_thinking is None else show_thinking,
            custom_renderer=getattr(self, "_custom_renderer", None),
            markdown_transformers=getattr(self, "_markdown_transformers", None),
            hidden_thinking_label=getattr(self, "_hidden_thinking_label", "Thinking"),
            tool_renderer=getattr(self, "_tool_renderer", None),
        )
        for label, text in entries:
            self.mount(MessageEntry(label, text))
        self.scroll_end(animate=False)

    def clear_messages(self) -> None:
        entries = list(self.query(MessageEntry)) + list(self.query(BashExecutionEntry))
        for entry in entries:
            entry.remove()


class PiEditor(TextArea):
    """多行输入编辑器：Enter 提交，Shift+Enter 插入换行。"""

    BINDINGS = [
        Binding("enter", "submit", "Send"),
        Binding("shift+enter", "newline", "Insert newline"),
        Binding("tab", "autocomplete", "Autocomplete"),
        Binding("ctrl+c", "copy_or_clear", "Copy or clear"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # slash 命令补全激活时：↑/↓ 导航、Enter 插入、Esc 关闭（编辑器保持焦点）。
        self.completion_active = False

    class AutocompleteRequested(Message):
        """Tab 按下且需要扩展自动补全。"""

        def __init__(self, editor: "PiEditor") -> None:
            super().__init__()
            self.editor = editor

    class Submitted(Message):
        """编辑器提交事件。"""

        def __init__(self, editor: "PiEditor", text: str) -> None:
            super().__init__()
            self.editor = editor
            self.text = text

    class ExitRequested(Message):
        """编辑器为空时按下 ctrl+d（退出快捷键）。"""

        pass

    class CopyRequested(Message):
        """ctrl+x：复制最后一条 assistant 消息（对齐 TS app.message.copy）。"""

        pass

    class CycleThinkingRequested(Message):
        """shift+tab：循环 thinking 级别（对齐 TS app.thinking.cycle）。"""

        pass

    class CompletionNavigateRequested(Message):
        """补全激活时 ↑/↓ 移动选中项。"""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class CompletionSubmitRequested(Message):
        """补全激活时 Enter 确认选中项。"""

        pass

    class CompletionHideRequested(Message):
        """补全激活时 Esc 关闭补全。"""

        pass

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.clear()
        self.post_message(self.Submitted(self, text))

    def action_newline(self) -> None:
        """Shift+Enter：插入换行（TextArea 会吞掉单独的 enter）。"""
        self._replace_via_keyboard("\n", *self.selection)

    def action_autocomplete(self) -> None:
        self.post_message(self.AutocompleteRequested(self))

    def action_copy_or_clear(self) -> None:
        """ctrl+c：有选区则复制选中文本，否则清空编辑器（对齐 TS）。"""
        if self.selected_text:
            self.action_copy()
        else:
            self.clear()

    async def _on_key(self, event: events.Key) -> None:
        # TextArea._on_key 会把 enter 直接当换行插入并 stop() 事件，
        # 导致上面的 "enter -> submit" 绑定永远不触发，必须在这里拦截。
        if self.completion_active:
            if event.key in ("up", "down"):
                self.post_message(self.CompletionNavigateRequested(-1 if event.key == "up" else 1))
                event.stop()
                event.prevent_default()
                return
            if event.key == "escape":
                self.post_message(self.CompletionHideRequested())
                event.stop()
                event.prevent_default()
                return
        if event.key == "enter":
            if self.completion_active:
                self.post_message(self.CompletionSubmitRequested())
            else:
                self.action_submit()
            event.stop()
            event.prevent_default()
            return
        if event.key == "ctrl+d":
            if not self.text:
                # 编辑器为空时 ctrl+d = 退出请求（TextArea 默认会把它当删除键吞掉，
                # 冒泡到应用绑定不可靠，改为显式发消息）。
                self.post_message(self.ExitRequested())
                event.stop()
                event.prevent_default()
                return
            # 非空：保留 TextArea 默认行为（删除右侧字符）。
            await super()._on_key(event)
            return
        if event.key == "ctrl+x":
            # TextArea 默认把 ctrl+x 当剪切；对齐 TS：ctrl+x 复制最后一条消息。
            self.post_message(self.CopyRequested())
            event.stop()
            event.prevent_default()
            return
        if event.key == "shift+tab":
            # Textual 默认用 shift+tab 切换焦点；对齐 TS：循环 thinking 级别。
            self.post_message(self.CycleThinkingRequested())
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)
        # 自动触发 slash 命令补全：输入 `/cmd` 且尚未出现空格时。
        character = getattr(event, "character", None)
        if (
            character
            and character.isprintable()
            and self.text.startswith("/")
            and " " not in self.text
        ):
            self.post_message(self.AutocompleteRequested(self))


class PiEditorVim(PiEditor):
    """vim 风格编辑器：Esc 切换 normal/insert，normal 模式支持移动与编辑。

    normal 模式快捷键：h/j/k/l 移动、0/$ 行首/行尾、i/a/o 进入插入、
    dd 删行、x 删字符、u 撤销；Enter 提交（与 PiEditor 一致）。
    """

    class ModeChanged(Message):
        """normal / insert 模式切换。"""

        def __init__(self, editor: "PiEditorVim", mode: str) -> None:
            super().__init__()
            self.editor = editor
            self.mode = mode

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vim_mode = "insert"
        self._pending: str = ""

    def toggle_mode(self) -> None:
        self.vim_mode = "normal" if self.vim_mode == "insert" else "insert"
        self._pending = ""
        self.post_message(self.ModeChanged(self, self.vim_mode))

    async def _on_key(self, event: events.Key) -> None:
        if self.vim_mode == "insert":
            await self._on_insert_key(event)
            return
        await self._on_normal_key(event)

    async def _on_insert_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.toggle_mode()
            return
        await super()._on_key(event)

    async def _on_normal_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.toggle_mode()
            return
        if key == "enter":
            self.action_submit()
            event.stop()
            event.prevent_default()
            return
        if key == "i":
            self.toggle_mode()
            event.stop()
            event.prevent_default()
            return
        if key == "a":
            self.move_cursor_relative(columns=1)
            self.toggle_mode()
            event.stop()
            event.prevent_default()
            return
        if key == "o":
            self._open_line_below()
            event.stop()
            event.prevent_default()
            return
        if key == "h":
            self.move_cursor_relative(columns=-1)
        elif key == "l":
            self.move_cursor_relative(columns=1)
        elif key == "j":
            self.move_cursor_relative(rows=1)
        elif key == "k":
            self.move_cursor_relative(rows=-1)
        elif key == "0":
            self.move_cursor((self.cursor_location[0], 0))
        elif key == "$":
            row = self.cursor_location[0]
            self.move_cursor((row, len(self.document.lines[row])))
        elif key == "x":
            self._delete_char()
        elif key == "u":
            self.undo()
        elif key == "d":
            if self._pending == "d":
                self._pending = ""
                self._delete_line()
            else:
                self._pending = "d"
            event.stop()
            event.prevent_default()
            return
        else:
            self._pending = ""
            return
        self._pending = ""
        event.stop()
        event.prevent_default()

    def _open_line_below(self) -> None:
        row = self.cursor_location[0]
        lines = self.document.lines
        if row >= len(lines):
            row = max(0, len(lines) - 1)
        self.insert("\n", (row, len(lines[row])))
        self.move_cursor((row + 1, 0))
        self.toggle_mode()

    def _delete_line(self) -> None:
        row = self.cursor_location[0]
        lines = self.document.lines
        if row >= len(lines):
            return
        if row == len(lines) - 1:
            self.delete((row, 0), (row, len(lines[row])))
        else:
            self.delete((row, 0), (row + 1, 0))

    def _delete_char(self) -> None:
        row, col = self.cursor_location
        lines = self.document.lines
        if row >= len(lines) or col >= len(lines[row]):
            return
        self.delete((row, col), (row, col + 1))


class PiStatusBar(Label):
    """状态栏：Working/Compaction/Retry/Idle。"""


class PiFooter(Label):
    """底部栏：模型 / 思考级别 / 消息数。"""

    def update_info(
        self,
        *,
        model: str,
        thinking: str,
        message_count: int,
        session_name: str | None = None,
    ) -> None:
        name = f" [{session_name}]" if session_name else ""
        self.update(f"model: {model}  thinking: {thinking}  messages: {message_count}{name}")


class PiToolbar(Input):
    """工具条输入（占位；后续用于搜索等）。"""


__all__ = [
    "BashExecutionEntry",
    "MessageEntry",
    "PiHeader",
    "PiChatContainer",
    "PiEditor",
    "PiEditorVim",
    "PiStatusBar",
    "PiFooter",
    "message_to_entries",
]
