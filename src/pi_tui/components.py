"""TUI 组件（引擎版）：Header / ChatContainer / Editor / StatusBar / Footer。"""

from __future__ import annotations

import base64
from typing import Any

from rich.style import Style

from pi_tui.engine.cells import Cell, Line, blank_line, line_from_text
from pi_tui.engine.text import DefaultTextStyle, render_markdown, render_markup, strip_ansi
from pi_tui.engine.widgets import (
    Editor,
    Input,
    Label,
    Message,
    ScrollView,
    Static,
    Vertical,
    Widget,
)

from .keybindings import KeybindingsManager
from .links import linkify_lines, linkify_paths, normalize_path_slashes
from .markdown import label_icon
from .terminal_image import (
    detect_capabilities,
    encode_iterm2_image,
    encode_kitty_delete,
    encode_kitty_placement,
)


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


def _collect_image_data(content: Any, images_out: list[bytes] | None) -> None:
    """收集内容块里的 image data（base64 字符串或 bytes）到 images_out。"""
    if images_out is None or not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        data = block.get("data")
        if isinstance(data, str):
            try:
                images_out.append(base64.b64decode(data))
            except Exception:
                pass
        elif isinstance(data, (bytes, bytearray)):
            images_out.append(bytes(data))


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
    skip_tool_calls: bool = False,
    images_out: list[bytes] | None = None,
) -> list[tuple[str, str]]:
    """AgentMessage → [(label, text)]，供聊天容器渲染。"""
    role = message.get("role")
    content = message.get("content")
    _collect_image_data(content, images_out)

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
                return [(label, linkify_paths(rendered))]
        if isinstance(content, str):
            return [(label, linkify_paths(content))]
        text = "\n".join(_block_text(block) for block in content or [])
        return [(label, linkify_paths(text))]

    if role == "assistant":
        entries: list[tuple[str, str]] = []
        label = "Assistant (error)" if message.get("error_message") else "Assistant"
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[str] = []
        has_thinking = False
        for block in content or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                has_thinking = True
                if show_thinking:
                    thinking_parts.append(block.get("thinking", ""))
            elif block_type == "toolCall":
                if show_tools and not skip_tool_calls:
                    name = block.get("name", "tool")
                    arguments = block.get("arguments") or {}
                    tool_calls.append(f"{name}({linkify_paths(str(arguments))})")
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
        elif has_thinking and not show_thinking:
            entries.append((hidden_thinking_label, ""))
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
        if not entries and images_out:
            entries.append((label, "[image]"))
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
                return [(custom_type, linkify_paths(rendered))]
        if isinstance(content, str):
            return [(custom_type, linkify_paths(content))]
        text = "\n".join(_block_text(block) for block in content or [])
        return [(custom_type, linkify_paths(text))] if text else []
    # 其它角色降级为文本。
    if isinstance(content, str):
        return [("Agent", linkify_paths(content))]
    text = "\n".join(_block_text(block) for block in content or [])
    return [("Agent", linkify_paths(text))] if text else []


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------


def _render_labeled_markdown(
    label: str,
    text: str,
    width: int,
    speaking: bool = False,
    prompt_marker: bool = False,
    theme_colors: dict | None = None,
    kind: str = "text",
) -> list[Line]:
    """消息条目渲染：图标 + label（粗体，单独一行）+ 缩进正文。"""
    suffix = " Speaking…" if speaking else ""
    label_line = line_from_text(f"{label_icon(label)} {label}{suffix}", width, Style(bold=True))
    body = normalize_path_slashes(text)
    default_style = None
    if kind == "thinking":
        default_style = DefaultTextStyle(
            color=(theme_colors or {}).get("thinking"),
            italic=True,
        )
    if "[/" not in body:
        body_lines = render_markdown(
            body,
            max(0, width - 2),
            theme_colors=theme_colors,
            default_style=default_style,
        )
    else:
        body_lines = []
        for raw_line in linkify_paths(body).splitlines() or [""]:
            body_lines.extend(render_markup(raw_line, max(0, width - 2)))
    lines: list[Line] = [label_line]
    for body_line in body_lines:
        # 直接构造前导两空格 + 定宽正文行（整行覆盖），
        # 让布局引擎可以整行复用，避免每帧逐格 patch。
        cells = ([Cell(" "), Cell(" ")] + list(body_line.cells))[:width]
        if len(cells) < width:
            cells.extend(Cell(" ") for _ in range(width - len(cells)))
        indented = Line(cells)
        lines.append(indented)
    linkify_lines(lines)
    if prompt_marker and lines:
        lines[0].passthrough = "\x1b]133;A\x07" + lines[0].passthrough
    return lines


class MessageEntry(Widget):
    """聊天消息条目（支持终端图像 placement passthrough）。"""

    def __init__(
        self,
        label: str,
        text: str,
        images: list[bytes] | None = None,
        *,
        image_width: int | None = None,
        kind: str = "text",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.entry_text = text
        self.kind = kind
        self.images = list(images or [])
        self.image_width = image_width
        self.prompt_marker = False
        self.theme_colors: dict = {}
        self._image_id_base = id(self) & 0xFFFFFF
        self.speaking = False
        # 内容版本缓存：流式更新只使当前条目失效，历史条目跨帧复用渲染结果。
        self._content_version = 0
        self._render_cache: dict[tuple[int, int], list[Line]] = {}
        self._size_cache: dict[int, int] = {}
        self._content_size_cache: tuple[int, int] | None = None

    @property
    def is_mounted(self) -> bool:
        return self.app is not None

    def set_text(self, text: str) -> None:
        """流式更新正文。"""
        if text == self.entry_text:
            return
        self.entry_text = text
        self._bump_version()
        self.refresh()

    def set_speaking(self, speaking: bool) -> None:
        """标记当前是否正在说话（流式回复中显示 Speaking…）。"""
        if speaking == self.speaking:
            return
        self.speaking = speaking
        self._bump_version()
        self.refresh()

    def _bump_version(self) -> None:
        self._content_version += 1
        self._render_cache.clear()
        self._size_cache.clear()
        self._content_size_cache = None

    def remove(self, child=None) -> None:
        """移除时清理已显示的 kitty 图片。"""
        if self.images and "kitty" in detect_capabilities() and self.app is not None:
            for index in range(len(self.images)):
                self.app.terminal.write(encode_kitty_delete(self._image_id_base + index))
        super().remove(child)

    def render(self, width: int, height: int) -> list[Line]:
        key = (width, height)
        cached = self._render_cache.get(key)
        if cached is not None:
            return cached
        lines = _render_labeled_markdown(
            self.label,
            self.entry_text,
            width,
            self.speaking,
            self.prompt_marker,
            self.theme_colors or None,
            self.kind,
        )
        if self.images:
            capabilities = detect_capabilities()
            if capabilities:
                while len(lines) < 2:
                    lines.append(blank_line(width, self.base_style))
                if "kitty" in capabilities:
                    lines[1].passthrough = "".join(
                        encode_kitty_placement(
                            image,
                            image_id=self._image_id_base + index,
                            width=self.image_width,
                            height=self.image_width,
                        )
                        for index, image in enumerate(self.images)
                    )
                elif "iterm2" in capabilities:
                    lines[1].passthrough = "".join(
                        encode_iterm2_image(image, name=f"image-{index}")
                        for index, image in enumerate(self.images)
                    )
        if len(lines) > height:
            lines = lines[:height]
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        for line in lines:
            line.shared = True
        self._render_cache[key] = lines
        return lines

    def handle_mouse(self, event) -> bool:
        if event.type == "release" and self.entry_text:
            self.post_message(CopyRequested(self.entry_text), "")
            return True
        return False

    def content_size(self) -> tuple[int, int]:
        if self._content_size_cache is None:
            lines = _render_labeled_markdown(
                self.label,
                self.entry_text,
                1000,
                self.speaking,
                False,
                self.theme_colors or None,
                self.kind,
            )
            self._content_size_cache = (1000, len(lines))
        return self._content_size_cache

    def natural_size(self, width: int) -> tuple[int, int]:
        """按实际内容宽度估算换行后的高度（长消息不再被截断）。"""
        content_width = max(1, int(width) - 2)
        height = self._size_cache.get(content_width)
        if height is None:
            lines = _render_labeled_markdown(
                self.label,
                self.entry_text,
                content_width,
                self.speaking,
                self.prompt_marker,
                self.theme_colors or None,
                self.kind,
            )
            height = len(lines)
            self._size_cache[content_width] = height
        return (max(1, int(width)), height)


class ToolExecutionEntry(Widget):
    """工具执行条目：名称/参数 + 输出 + 状态，可展开/折叠（对齐 TS ToolExecutionComponent）。"""

    _STATUS_BG: dict[str, str] = {
        "running": "toolPendingBg",
        "success": "toolSuccessBg",
        "error": "toolErrorBg",
    }

    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Any = None,
        *,
        render_call=None,
        render_result=None,
        render_theme=None,
        theme_colors: dict | None = None,
        image_width: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.arguments = arguments
        self.render_call = render_call
        self.render_result = render_result
        self.render_theme = render_theme
        self.theme_colors = dict(theme_colors or {})
        self.image_width = image_width
        self.result_payload: Any = None
        self.output = ""
        self.status = "running"  # running / success / error
        self.expanded = False
        self.images: list[bytes] = []
        self._image_id_base = id(self) & 0xFFFFFF

    def update_arguments(self, arguments) -> None:
        self.arguments = arguments
        self.refresh()

    def set_partial_result(self, output: str, result: Any = None) -> None:
        """流式局部结果：状态保持 running，输出逐步更新。"""
        self.output = output
        self.result_payload = result
        self.status = "running"
        self._set_images(result)
        self.refresh()

    def set_result(self, output: str, is_error: bool = False, result: Any = None) -> None:
        self.output = output
        self.result_payload = result
        self.status = "error" if is_error else "success"
        self._set_images(result)
        self.refresh()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.refresh()

    def _set_images(self, result: Any) -> None:
        self.images = []
        content = None
        if isinstance(result, dict):
            content = result.get("content")
        elif result is not None:
            content = getattr(result, "content", None)
        _collect_image_data(content, self.images)

    def remove(self, child=None) -> None:
        """移除时清理已显示的 kitty 图片。"""
        if self.images and "kitty" in detect_capabilities() and self.app is not None:
            for index in range(len(self.images)):
                self.app.terminal.write(encode_kitty_delete(self._image_id_base + index))
        super().remove(child)

    def handle_mouse(self, event) -> bool:
        if event.type == "release":
            self.expanded = not self.expanded
            self.refresh()
            return True
        return False

    def _format_arguments(self) -> str:
        if isinstance(self.arguments, dict):
            return ", ".join(f"{key}={value}" for key, value in self.arguments.items())
        return str(self.arguments or "")

    def _render_call_text(self) -> str:
        if self.render_call is None:
            return self._format_arguments()
        rendered = self.render_call(self.arguments, self.render_theme, None)
        if isinstance(rendered, list):
            return "\n".join(str(line) for line in rendered)
        return str(rendered)

    def _render_result_text(self) -> str:
        if self.render_result is None or self.result_payload is None:
            return self.output
        rendered = self.render_result(
            self.result_payload,
            {"is_error": self.status == "error"},
            self.render_theme,
            None,
        )
        if isinstance(rendered, list):
            return "\n".join(str(line) for line in rendered)
        return str(rendered)

    def content_size(self) -> tuple[int, int]:
        lines = 1 + len(self.images)
        if self.expanded and self.output:
            lines += len(str(self.output).splitlines())
        return (1000, lines)

    def render(self, width: int, height: int) -> list[Line]:
        status = {"running": "...", "success": "ok", "error": "error"}[self.status]
        label = f"Tool: {self.tool_name} ({status})"
        bg_key = self._STATUS_BG[self.status]
        label_style = Style(
            color=self.theme_colors.get("toolTitle"),
            bgcolor=self.theme_colors.get(bg_key),
        )
        lines = [line_from_text(f"{label}  {self._render_call_text()}", width, label_style)]
        result_text = self._render_result_text()
        if self.expanded and result_text:
            output_style = Style(color=self.theme_colors.get("toolOutput"))
            for raw in str(result_text).splitlines():
                body = blank_line(width)
                body.patch(2, line_from_text(raw, max(0, width - 2), output_style))
                lines.append(body)
        if self.images:
            capabilities = detect_capabilities()
            if capabilities:
                while len(lines) < 2:
                    lines.append(blank_line(width, self.base_style))
                if "kitty" in capabilities:
                    lines[1].passthrough = "".join(
                        encode_kitty_placement(
                            image,
                            image_id=self._image_id_base + index,
                            width=self.image_width,
                            height=self.image_width,
                        )
                        for index, image in enumerate(self.images)
                    )
                elif "iterm2" in capabilities:
                    lines[1].passthrough = "".join(
                        encode_iterm2_image(image, name=f"tool-image-{index}")
                        for index, image in enumerate(self.images)
                    )
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        return lines[:height]


class BashExecutionEntry(Widget):
    """交互 bash 命令条目：流式输出 + 完成状态 + 展开预览（对齐 TS BashExecutionComponent）。"""

    PREVIEW_LINES = 20

    def __init__(
        self,
        command: str,
        *,
        exclude_from_context: bool = False,
        theme_colors: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.command = command
        self.exclude_from_context = exclude_from_context
        self.theme_colors = dict(theme_colors or {})
        self.output_lines: list[str] = []
        self.output = ""
        self.status: str = "running"  # running / complete / cancelled / error
        self.exit_code: int | None = None
        self.cancelled = False
        self.truncated = False
        self.full_output_path: str | None = None
        self.expanded = False

    @property
    def is_mounted(self) -> bool:
        return self.app is not None

    @property
    def label(self) -> str:
        return "Bash (excluded)" if self.exclude_from_context else "Bash"

    def append_output(self, chunk: str) -> None:
        clean = strip_ansi(chunk).replace("\r\n", "\n").replace("\r", "\n")
        new_lines = clean.split("\n")
        if self.output_lines and new_lines:
            self.output_lines[-1] += new_lines[0]
            self.output_lines.extend(new_lines[1:])
        else:
            self.output_lines.extend(new_lines)
        self.output = "\n".join(self.output_lines)
        self.refresh()

    def set_complete(
        self,
        exit_code: int | None,
        cancelled: bool = False,
        truncated: bool = False,
        full_output_path: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.cancelled = bool(cancelled)
        self.truncated = bool(truncated)
        self.full_output_path = full_output_path
        if self.cancelled:
            self.status = "cancelled"
        elif exit_code not in (None, 0):
            self.status = "error"
        else:
            self.status = "complete"
        self.refresh()

    def set_error(self, message: str) -> None:
        self.status = "error"
        self.full_output_path = None
        self.output_lines.append(f"(failed: {message})")
        self.output = "\n".join(self.output_lines)
        self.refresh()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.refresh()

    def _status_parts(self) -> list[str]:
        parts: list[str] = []
        if self.status == "running":
            parts.append("Running...")
            return parts
        available = self.output_lines
        display = available if self.expanded else available[-self.PREVIEW_LINES :]
        hidden = len(available) - len(display)
        if hidden > 0:
            parts.append(
                "(to collapse)" if self.expanded else f"... {hidden} more lines (to expand)"
            )
        if self.status == "cancelled":
            parts.append("(cancelled)")
        elif self.status == "error":
            failed = [line for line in self.output_lines if line.startswith("(failed:")]
            if failed:
                parts.append(failed[-1])
            else:
                parts.append(f"(exit {self.exit_code})")
        if self.truncated and self.full_output_path:
            parts.append(f"Output truncated. Full output: {self.full_output_path}")
        return parts

    def render(self, width: int, height: int) -> list[Line]:
        color_key = "dim" if self.exclude_from_context else "bashMode"
        border_color = self.theme_colors.get(color_key)
        border_style = Style(color=border_color)
        lines: list[Line] = []
        lines.append(line_from_text("\u2500" * width, width, border_style))
        lines.append(
            line_from_text("$ " + self.command, width, Style(color=border_color, bold=True))
        )
        available = self.output_lines
        display = available if self.expanded else available[-self.PREVIEW_LINES :]
        output_style = Style(color=self.theme_colors.get("toolOutput"))
        if display:
            lines.append(blank_line(width, None))
            for raw in display:
                lines.append(line_from_text(raw, width, output_style))
        status_parts = self._status_parts()
        if status_parts:
            lines.append(blank_line(width, None))
            for part in status_parts:
                lines.append(line_from_text(part, width, border_style))
        lines.append(line_from_text("\u2500" * width, width, border_style))
        while len(lines) < height:
            lines.append(blank_line(width, self.base_style))
        return lines[:height]

    def handle_mouse(self, event) -> bool:
        if event.type == "release":
            parts = [f"$ {self.command}", "\n".join(self.output_lines)]
            self.post_message(CopyRequested("\n".join(parts)), "")
            return True
        return False

    def content_size(self) -> tuple[int, int]:
        height = 3  # 上下边框 + 命令行
        available = self.output_lines
        display = available if self.expanded else available[-self.PREVIEW_LINES :]
        if display:
            height += 1 + len(display)
        status_parts = self._status_parts()
        if status_parts:
            height += 1 + len(status_parts)
        return (1000, height)


class PiHeader(Static):
    """Logo + 快捷键提示（紧凑/展开两种形态，对齐 TS builtInHeader）。"""

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
        # 对齐 TS transcriptScrollView：自动跟随底部、主滚动视口、overscroll chain。
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
        self,
        *,
        show_images: bool = True,
        image_width_cells: int | None = None,
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
        """条目挂载进内部消息列表（而非滚动视口本身）。"""
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
        """把指定条目滚入视口。"""
        row, _col, _w, _h = widget.rect
        if row < self.scroll_offset:
            self.scroll_offset = row
        elif row >= self.scroll_offset + self.rect[3]:
            self.scroll_offset = max(0, row - self.rect[3] + 1)
        self.refresh()


class PiEditor(Editor):
    """多行输入编辑器（引擎版）。"""


class PiEditorVim(PiEditor):
    """vim 风格编辑器（引擎版）。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vim_enabled = True


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
