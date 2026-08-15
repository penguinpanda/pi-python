"""TUI 扩展 UI 上下文（对齐 TS 的 TUI UIContext）。"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

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

    def strikethrough(self, text: str = "") -> str:
        return f"\x1b[9m{text}\x1b[0m" if text else "\x1b[9m"


class CustomDialog:
    """把扩展 factory 返回的普通 Widget 包装成可关闭的 overlay 对话框。"""

    def __init__(self, inner: Any, resolve: Callable[[Any], None]) -> None:
        self._inner = inner
        self._resolve = resolve
        self._closed = False

    @property
    def app(self) -> Any:
        return self._inner.app

    @app.setter
    def app(self, value: Any) -> None:
        self._inner.app = value

    def handle_key(self, key) -> bool:
        handler = getattr(self._inner, "handle_key", None)
        if handler is None:
            return False
        return bool(handler(key))

    def render(self, width: int, height: int):
        renderer = getattr(self._inner, "render", None)
        if renderer is None:
            from pi_tui.engine.cells import blank_line

            return [blank_line(width) for _ in range(max(0, height))]
        return renderer(width, height)

    def content_size(self):
        method = getattr(self._inner, "content_size", None)
        if method is None:
            return (0, 0)
        return method()

    def natural_size(self, width: int):
        method = getattr(self._inner, "natural_size", None)
        if method is None:
            return self.content_size()
        return method(width)

    def dismiss(self, value: Any = None) -> None:
        if self._closed:
            return
        self._closed = True
        app = self._inner.app
        if app is not None and hasattr(app, "_close_overlay_dialog"):
            app._close_overlay_dialog(self, value)
        else:
            self._resolve(value)


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

        selector = ChoiceSelector(title, list(options))
        self._app.push_screen(selector, callback=_callback)
        countdown_task = self._start_countdown(selector, title, timeout)
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            if countdown_task is not None:
                countdown_task.cancel()

    def _start_countdown(self, dialog, title: str, timeout: float | None):
        """可见倒计时（对齐 TS CountdownTimer：对话框内显示 auto-cancel in Xs）。"""
        import math
        import time as time_mod

        if timeout is None:
            return None
        deadline = time_mod.monotonic() + timeout

        async def _countdown() -> None:
            while True:
                remaining = max(0.0, deadline - time_mod.monotonic())
                dialog.update_title(f"{title}  (auto-cancel in {math.ceil(remaining)}s)")
                if remaining <= 0:
                    return
                await asyncio.sleep(1)

        return asyncio.create_task(_countdown())

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

        dialog = TextInputDialog(title, placeholder or "")
        self._app.push_screen(dialog, callback=_callback)
        countdown_task = self._start_countdown(dialog, title, timeout)
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            if countdown_task is not None:
                countdown_task.cancel()

    def get_theme(self, name: str):
        """按名加载主题（对齐 TS getTheme）。"""
        try:
            return self._app._theme_loader.load(name)
        except Exception:
            return None

    def get_all_themes(self) -> list[dict]:
        """全部可用主题（对齐 TS getAllThemes）。"""
        loader = self._app._theme_loader
        theme_dir = getattr(loader, "_theme_dir", None)
        return [
            {"name": name, "path": str(theme_dir / f"{name}.json") if theme_dir else None}
            for name in loader.available()
        ]

    def on_terminal_input(self, handler):
        """订阅终端输入流（对齐 TS onTerminalInput；返回取消订阅函数）。"""
        return self._app.add_terminal_input_handler(handler)

    def get_context_usage(self):
        """当前上下文 usage 汇总（对齐 TS getContextUsage）。"""
        try:
            stats = self._app._session.get_session_stats()
        except Exception:
            return None
        tokens = stats.get("tokens") or {}
        return {
            "inputTokens": tokens.get("input", 0),
            "outputTokens": tokens.get("output", 0),
            "totalTokens": tokens.get("total", 0),
            "costUsd": stats.get("cost", 0),
        }

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

    def set_widget(self, key: str, lines: list[str] | None, options: dict | None = None) -> None:
        """在编辑器上方（默认）或下方显示多行组件。"""
        self._app._set_widget(key, list(lines or []), options or {})

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

    async def custom(
        self,
        factory,
        *,
        overlay_options: dict | None = None,
        on_handle=None,
    ) -> Any:
        """显示扩展自定义交互组件（对齐 TS ctx.ui.custom）。

        factory(tui, theme, keybindings, done) 返回一个 Widget；
        组件通过调用 done(result) 提交结果并关闭。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        holder: dict[str, Any] = {}

        def resolve(value: Any) -> None:
            if not future.done():
                future.set_result(value)

        def done(value: Any) -> None:
            dialog = holder.get("dialog")
            if dialog is not None:
                dialog.dismiss(value)
            else:
                resolve(value)

        inner = factory(self._app, self.theme, self._app._keybindings, done)
        inner.app = self._app
        dialog = CustomDialog(inner, resolve)
        holder["dialog"] = dialog
        # push_screen 使用默认 overlay 布局；overlay_options 预留给后续
        # push_screen 支持自定义布局时使用。
        self._app.push_screen(dialog, callback=done)
        if on_handle is not None:
            on_handle(self._app._overlay_manager.entry_for_widget(dialog))
        try:
            return await future
        finally:
            if not future.done():
                future.cancel()
                # 消费取消异常,避免无人 retrieve 的 Future 日志。
                future.exception()
