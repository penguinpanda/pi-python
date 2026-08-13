"""交互模式对话框组件。"""

from __future__ import annotations

from typing import Any, Callable

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import Input, SelectItem, SelectList, Widget


class LoginDialogComponent(Widget):
    """OAuth 登录对话框（对齐 TS LoginDialogComponent 的用户可见能力）。"""

    def __init__(
        self,
        provider_id: str,
        on_complete: Callable[[bool, str | None], None],
        provider_name: str | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__()
        self.provider_id = provider_id
        self._on_complete = on_complete
        self._provider_name = provider_name or provider_id
        self._title = title or f"Login to {self._provider_name}"
        self._lines: list[str] = []
        self._input = Input(value="", placeholder="")
        self._input_resolver: Callable[[str], None] | None = None
        self._input_rejecter: Callable[[Exception], None] | None = None
        self._input_visible = False
        self._prompt: str | None = None
        self._hint = "Esc to cancel"

    def _clear_content(self) -> None:
        self._lines = []
        self._input_visible = False
        self._prompt = None
        self._input_resolver = None
        self._input_rejecter = None
        self.refresh()

    def _add(self, *lines: str) -> None:
        self._lines.extend(lines)

    def _hyperlink(self, url: str, label: str) -> str:
        return f"\x1b]8;;{url}\x07{label}\x1b]8;;\x07"

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        self._clear_content()
        self._add(self._hyperlink(url, url))
        self._add(self._hyperlink(url, "Ctrl+click to open"))
        if instructions:
            self._add(instructions)
        self.refresh()

    def show_device_code(self, info: dict[str, Any]) -> None:
        self._clear_content()
        verification_uri = str(info.get("verificationUri") or info.get("verification_uri") or "")
        user_code = str(info.get("userCode") or info.get("user_code") or "")
        if verification_uri:
            self._add(self._hyperlink(verification_uri, verification_uri))
        self._add(f"Enter code: {user_code}")
        self.refresh()

    def show_manual_input(self, prompt: str) -> str | None:
        self._prompt = prompt
        self._input_visible = True
        self._input.value = ""
        self.refresh()
        return None

    def show_prompt(self, message: str, placeholder: str | None = None) -> str | None:
        self._prompt = message
        self._input_visible = True
        self._input.value = ""
        if placeholder:
            self._input.placeholder = placeholder
        self.refresh()
        return None

    def show_details(self, lines: list[str]) -> None:
        self._clear_content()
        self._add(*lines)
        self.refresh()

    def show_info(self, message: str, links: list[dict[str, str]] | None = None) -> None:
        self._add(message)
        for link in links or []:
            url = link.get("url", "")
            label = link.get("label") or url
            self._add(self._hyperlink(url, label))
        self.refresh()

    def show_waiting(self, message: str) -> None:
        self._add(message)
        self._add("Esc to cancel")
        self.refresh()

    def show_progress(self, message: str) -> None:
        self._add(message)
        self.refresh()

    def _submit_input(self) -> None:
        if not self._input_visible:
            return
        value = self._input.value
        self._input_visible = False
        self._input_resolver, resolver = None, self._input_resolver
        self._input_rejecter = None
        if resolver is not None:
            resolver(value)
        self.refresh()

    def _cancel(self) -> None:
        if self._input_visible and self._input_rejecter is not None:
            rejecter = self._input_rejecter
            self._input_resolver = None
            self._input_rejecter = None
            self._input_visible = False
            rejecter(RuntimeError("Login cancelled"))
            self.refresh()
            return
        self._on_complete(False, "Login cancelled")

    def set_prompt_result(self, value: str) -> None:
        if self._input_resolver is not None:
            resolver = self._input_resolver
            self._input_resolver = None
            self._input_rejecter = None
            self._input_visible = False
            resolver(value)
            self.refresh()

    def handle_key(self, key: Key) -> bool:
        if key.name == "escape":
            self._cancel()
            return True
        if key.name == "enter" and self._input_visible:
            self._submit_input()
            return True
        if self._input_visible:
            return self._input.handle_key(key)
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines = [line_from_text(self._title, width)]
        for raw in self._lines:
            lines.append(line_from_text(raw, width))
        if self._input_visible:
            if self._prompt:
                lines.append(line_from_text(self._prompt, width))
            lines.extend(self._input.render(width, 1))
            lines.append(line_from_text(self._hint, width))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (60, min(2 + len(self._lines) + (2 if self._input_visible else 0), 12))


class ShowImagesSelectorComponent(Widget):
    """是否内联显示图片的选择器。"""

    def __init__(
        self,
        current_value: bool,
        on_select: Callable[[bool], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_select = on_select
        self._on_cancel = on_cancel
        items = [
            SelectItem(value="yes", label="Yes", description="Show images inline in terminal"),
            SelectItem(value="no", label="No", description="Show text placeholder instead"),
        ]
        self._list = SelectList(
            items,
            current="yes" if current_value else "no",
            enable_search=False,
            max_height=4,
        )

    def handle_key(self, key: Key) -> bool:
        if key.name == "enter":
            item = self._list.selected_item
            if item is not None:
                self._on_select(item.value == "yes")
            else:
                self._on_cancel()
            return True
        if key.name == "escape":
            self._on_cancel()
            return True
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._list.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._list.content_size()
