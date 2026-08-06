"""TUI 扩展 UI 上下文（对齐 TS 的 TUI UIContext）。"""

from __future__ import annotations

import asyncio

from pi_tui.selectors import ChoiceSelector, TextInputDialog


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    hex_value = value.lstrip("#")
    if len(hex_value) != 6:
        return (0, 0, 0)
    try:
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
    except ValueError:
        return (0, 0, 0)
    return (r, g, b)


class ThemeFacade:
    """ctx.ui.theme：按主题色名生成 ANSI 前景 / 背景色。"""

    def __init__(self, theme) -> None:
        self._colors = theme.colors

    def fg(self, name: str, text: str = "") -> str:
        color = self._colors.get(name, "")
        if not color:
            return text
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m" if text else f"\x1b[38;2;{r};{g};{b}m"

    def bg(self, name: str, text: str = "") -> str:
        color = self._colors.get(name, "")
        if not color:
            return text
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[48;2;{r};{g};{b}m{text}\x1b[0m" if text else f"\x1b[48;2;{r};{g};{b}m"


class TuiUIContext:
    """PiTuiApp 的扩展 UI 实现。"""

    def __init__(self, app) -> None:
        self._app = app

    @property
    def theme(self) -> ThemeFacade:
        return ThemeFacade(self._app._theme)

    async def select(
        self, title: str, options: list[str], timeout: float | None = None
    ) -> str | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def _callback(value) -> None:
            if not future.done():
                future.set_result(value)

        self._app.push_screen(ChoiceSelector(title, list(options)), callback=_callback)
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None

    async def confirm(self, title: str, message: str, timeout: float | None = None) -> bool:
        choice = await self.select(f"{title}\n\n{message}", ["Yes", "No"], timeout=timeout)
        return choice == "Yes"

    async def input(
        self,
        title: str,
        placeholder: str | None = None,
        timeout: float | None = None,
    ) -> str | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def _callback(value) -> None:
            if not future.done():
                future.set_result(value)

        self._app.push_screen(
            TextInputDialog(title, placeholder or ""),
            callback=_callback,
        )
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None

    def notify(self, message: str, notify_type: str | None = None) -> None:
        self._app._set_status(message)

    def set_status(self, key: str, text: str | None) -> None:
        if text is not None:
            self._app._set_status(text)

    def set_title(self, title: str) -> None:
        self._app.set_title(title)

    def set_editor_text(self, text: str) -> None:
        self._app._editor.text = text

    def get_editor_text(self) -> str:
        return self._app._editor.text

    def paste_to_editor(self, text: str) -> None:
        """把文本插入编辑器（对齐 TS pasteToEditor）。"""
        try:
            self._app._editor.insert(text)
        except Exception:
            self._app._editor.text += text

    async def editor(self, title: str, prefill: str = "") -> str | None:
        """弹出文本编辑对话框（对齐 TS editor(title, prefill)）。"""
        return await self._app._await_text_input(title, prefill)

    def set_footer(self, text: str | None) -> None:
        self._app._footer.update(text or "")

    def set_header(self, text: str | None) -> None:
        self._app._header.update(text or "")

    def set_editor_component(self, component) -> None:
        """用自定义编辑器组件替换 PiEditor。"""
        self._app._replace_editor(component)

    def set_widget(self, key: str, lines: list[str], options: dict | None = None) -> None:
        """在编辑器上方（默认）或下方显示多行组件。"""
        self._app._set_widget(key, list(lines), options or {})

    def set_overlay(self, key: str, lines: list[str], options: dict | None = None) -> None:
        """显示浮层（锚点 + margin）。"""
        self._app._set_overlay(key, list(lines), options or {})

    def set_overlay_component(self, key: str, component, options: dict | None = None) -> None:
        """用任意组件作为 overlay（组件树 API）。"""
        self._app._set_overlay_component(key, component, options or {})

    def set_overlay_renderer(self, key: str, renderer, options: dict | None = None) -> None:
        """用渲染回调生成 overlay 内容：fn(width, height) -> list[str]。"""
        self._app._set_overlay_renderer(key, renderer, options or {})

    def set_hidden_thinking_label(self, label: str | None = None) -> None:
        """设置折叠 thinking 块的标签（None 恢复默认）。"""
        self._app._set_hidden_thinking_label(label)

    def set_working_message(self, text: str | None = None) -> None:
        """设置流式工作提示文案（None 恢复默认）。"""
        self._app._set_working_message(text)

    def set_working_visible(self, visible: bool) -> None:
        """控制 agent 运行时是否显示 Working 状态（对齐 TS setWorkingVisible）。"""
        self._app._working_visible = bool(visible)

    def set_working_indicator(self, options: dict | None = None) -> None:
        """配置 Working 指示器（message/visible，对齐 TS setWorkingIndicator）。"""
        options = options or {}
        message = options.get("message")
        if isinstance(message, str) and message:
            self._app._set_working_message(message)
        visible = options.get("visible")
        if isinstance(visible, bool):
            self._app._working_visible = visible

    def set_theme(self, theme: str | None = None) -> None:
        """切换主题（None 恢复当前配置主题）。"""
        self._app._set_theme(theme)
