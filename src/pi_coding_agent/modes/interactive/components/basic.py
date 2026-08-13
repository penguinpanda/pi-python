"""coding-agent 应用级 TUI 基础组件。"""

from __future__ import annotations

from typing import Any

from pi_tui.components import MessageEntry, message_to_entries
from pi_tui.engine.widgets import Editor, Input, Label, ScrollView, Static, Vertical, Widget
from pi_tui.keybindings import KeybindingsManager


class PiHeader(Static):
    """Logo + 快捷键提示。"""

    def __init__(self, keybindings: KeybindingsManager, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._keybindings = keybindings
        self._expanded = False
        self.refresh_hints()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.refresh_hints()

    def refresh_hints(self) -> None:
        model_forward = self._keybindings.get_action_key("app.model.cycleForward")
        model_select = self._keybindings.get_action_key("app.model.select")
        exit_key = self._keybindings.get_action_key("app.exit")
        if self._expanded:
            self.update(
                "[b]pi[/b]\n"
                f"interrupt {self._keybindings.get_action_key('app.interrupt') or 'escape'}\n"
                f"clear {self._keybindings.get_action_key('app.clear') or 'ctrl+c'}\n"
                f"exit {exit_key or 'ctrl+d'}\n"
                f"cycle model {model_forward or 'ctrl+p'}\n"
                f"select model {model_select or 'ctrl+l'}\n"
                "/  commands    !  bash    !!  bash (no context)\n"
                "drop files  to attach"
            )
        else:
            self.update(
                "[b]pi[/b]  "
                f"cycle model {model_forward or ''}  "
                f"select model {model_select or ''}  "
                f"exit {exit_key or ''}"
            )


class PiChatContainer(ScrollView):
    """消息列表容器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            Vertical(),
            follow="end",
            primary=True,
            overscroll="chain",
            **kwargs,
        )
        self._show_tools = True
        self._show_thinking = True
        self._custom_renderer = None
        self._markdown_transformers = None
        self._hidden_thinking_label = "Thinking"
        self._tool_renderer = None
        self._show_images = True
        self._image_width_cells: int | None = None
        self._theme_colors: dict = {}

    def set_image_options(
        self, *, show_images: bool = True, image_width_cells: int | None = None
    ) -> None:
        self._show_images = show_images
        self._image_width_cells = image_width_cells
        self.refresh()

    def set_theme_colors(self, colors: dict) -> None:
        self._theme_colors = dict(colors)
        self.refresh()

    def _body(self) -> Vertical:
        body = self.child
        assert isinstance(body, Vertical)
        return body

    def mount(self, child: Widget) -> Widget:
        return self._body().mount(child)

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
        skip_tool_calls: bool = False,
    ) -> None:
        images: list[bytes] = []
        entries = message_to_entries(
            message,
            show_tools=self._show_tools if show_tools is None else show_tools,
            show_thinking=self._show_thinking if show_thinking is None else show_thinking,
            custom_renderer=getattr(self, "_custom_renderer", None),
            markdown_transformers=getattr(self, "_markdown_transformers", None),
            hidden_thinking_label=getattr(self, "_hidden_thinking_label", "Thinking"),
            tool_renderer=getattr(self, "_tool_renderer", None),
            skip_tool_calls=skip_tool_calls,
            images_out=images,
        )
        for label, text in entries:
            kind = "thinking" if label == self._hidden_thinking_label else "text"
            self.mount(
                MessageEntry(
                    label,
                    text,
                    images=images if self._show_images else [],
                    image_width=self._image_width_cells,
                    kind=kind,
                )
            )
            if entries:
                last = self._body().children[-1]
                if isinstance(last, MessageEntry):
                    last.prompt_marker = message.get("role") == "user"
                    last.theme_colors = dict(self._theme_colors)
        self.scroll_end()

    def clear_messages(self) -> None:
        for entry in list(self._body().children):
            self._body().remove(entry)
        self.scroll_offset = 0
        self.refresh()

    def query(self, widget_type: type) -> list[Widget]:
        return [widget for widget in self._body().walk() if isinstance(widget, widget_type)]

    def scroll_to_widget(self, widget: Widget) -> None:
        row, _col, _w, _h = widget.rect
        if row < self.scroll_offset:
            self.scroll_offset = row
        elif row >= self.scroll_offset + self.rect[3]:
            self.scroll_offset = max(0, row - self.rect[3] + 1)
        self.refresh()


class PiEditor(Editor):
    """多行输入编辑器。"""


class PiEditorVim(PiEditor):
    """vim 风格编辑器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vim_enabled = True


class PiStatusBar(Label):
    """状态栏。"""


class PiFooter(Label):
    """底部栏。"""

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
    """工具条输入。"""


__all__ = [
    "PiHeader",
    "PiChatContainer",
    "PiEditor",
    "PiEditorVim",
    "PiStatusBar",
    "PiFooter",
    "PiToolbar",
]
