"""TUI 组件：Header / ChatContainer / Editor / StatusBar / Footer。"""

from __future__ import annotations

from typing import Any

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Input, Label, Static, TextArea

from .keybindings import KeybindingsManager


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


def message_to_entries(
    message: dict[str, Any],
    *,
    show_tools: bool = True,
    show_thinking: bool = True,
) -> list[tuple[str, str]]:
    """AgentMessage → [(label, text)]，供聊天容器渲染。"""
    role = message.get("role")
    content = message.get("content")

    if role == "user":
        if isinstance(content, str):
            return [("User", content)]
        text = "\n".join(_block_text(block) for block in content or [])
        return [("User", text)]

    if role == "toolResult":
        tool_name = message.get("tool_name", "tool")
        if isinstance(content, str):
            return [(f"Tool: {tool_name}", content)]
        text = "\n".join(_block_text(block) for block in content or [])
        return [(f"Tool: {tool_name}", text)]

    if role == "assistant":
        entries: list[tuple[str, str]] = []
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_parts: list[str] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                if show_thinking:
                    thinking_parts.append(block.get("thinking", ""))
            elif block_type == "toolCall":
                if show_tools:
                    tool_parts.append(_block_text(block))
            elif block_type == "text":
                text_parts.append(block.get("text", ""))
        if thinking_parts:
            entries.append(("Thinking", "\n".join(thinking_parts)))
        if text_parts:
            entries.append(("Assistant", "\n".join(text_parts)))
        if tool_parts:
            entries.append(("Tool call", "\n".join(tool_parts)))
        return entries

    if role == "compactionSummary":
        return [("Compaction", message.get("summary", ""))]
    if role == "system":
        return [("System", message.get("content", ""))]
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

    def on_mount(self) -> None:
        self.update(f"[b]{self.label}[/b] {self.entry_text}")


class PiHeader(Static):
    """Logo + 快捷键提示。"""

    def __init__(self, keybindings: KeybindingsManager, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._keybindings = keybindings

    def on_mount(self) -> None:
        model_forward = self._keybindings.get_action_key("app.model.cycleForward")
        model_select = self._keybindings.get_action_key("app.model.select")
        exit_key = self._keybindings.get_action_key("app.exit")
        self.update(
            "[b]pi[/b]  "
            f"cycle model {model_forward or ''}  "
            f"select model {model_select or ''}  "
            f"exit {exit_key or ''}"
        )


class PiChatContainer(VerticalScroll):
    """消息列表容器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._show_tools = True
        self._show_thinking = True

    def set_visibility(self, *, show_tools: bool, show_thinking: bool) -> None:
        self._show_tools = show_tools
        self._show_thinking = show_thinking

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
        )
        for label, text in entries:
            self.mount(MessageEntry(label, text))
        self.scroll_end(animate=False)

    def clear_messages(self) -> None:
        for entry in self.query(MessageEntry):
            entry.remove()


class PiEditor(TextArea):
    """多行输入编辑器：Enter 提交。"""

    BINDINGS = [
        Binding("enter", "submit", "Send"),
    ]

    class Submitted(Message):
        """编辑器提交事件。"""

        def __init__(self, editor: "PiEditor", text: str) -> None:
            super().__init__()
            self.editor = editor
            self.text = text

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.clear()
        self.post_message(self.Submitted(self, text))


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
        self.update(
            f"model: {model}  thinking: {thinking}  messages: {message_count}{name}"
        )


class PiToolbar(Input):
    """工具条输入（占位；后续用于搜索等）。"""


__all__ = [
    "MessageEntry",
    "PiHeader",
    "PiChatContainer",
    "PiEditor",
    "PiStatusBar",
    "PiFooter",
    "message_to_entries",
]
