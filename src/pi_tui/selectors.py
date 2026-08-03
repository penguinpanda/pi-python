"""选择器组件（3.6）：ModelSelector / SessionPicker。"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView


def _model_label(model) -> str:
    return f"{model.provider}/{model.id}  [dim]{model.name}[/dim]"


class ModelSelector(ModalScreen):
    """模型选择器：分组显示 + 实时搜索 + 键盘导航（Ctrl+L）。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, models: list[Any], current: Any | None = None) -> None:
        super().__init__()
        self._models = list(models)
        self._current = current
        self._query = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Select model", classes="selector-title")
            yield Input(
                placeholder="Search models...",
                value=self._query,
                classes="selector-input",
            )
            yield ListView(id="model-list")

    def on_mount(self) -> None:
        self._rebuild()

    def _filtered(self) -> list[Any]:
        query = self._query.lower()
        if not query:
            return self._models
        return [
            model
            for model in self._models
            if query in model.id.lower()
            or query in model.provider.lower()
            or query in (model.name or "").lower()
        ]

    def _rebuild(self) -> None:
        try:
            list_view = self.query_one("#model-list", ListView)
        except NoMatches:
            # 子组件尚未挂载完成（负载高时的竞态）：延迟重试。
            self.call_after_refresh(self._rebuild)
            return
        list_view.clear()
        current_key = (
            f"{self._current.provider}/{self._current.id}" if self._current else None
        )
        for model in self._filtered():
            key = f"{model.provider}/{model.id}"
            marker = ">" if key == current_key else " "
            list_view.append(
                ListItem(
                    Label(f"{marker} {_model_label(model)}"),
                    id=key.replace("/", "__").replace(".", "_"),
                )
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        self._query = event.value
        self._rebuild()

    def action_select(self) -> None:
        list_view = self.query_one("#model-list", ListView)
        if list_view.index is None:
            return
        filtered = self._filtered()
        index = list_view.index
        if 0 <= index < len(filtered):
            self.dismiss(filtered[index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionPicker(ModalScreen):
    """会话恢复选择器（--resume）：按修改时间排序。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        # [{path, session_id, modified}]
        self._sessions = list(sessions)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Resume session", classes="selector-title")
            list_view = ListView(id="session-list")
            for session in self._sessions:
                from datetime import datetime

                when = datetime.fromtimestamp(session["modified"]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                list_view.append(
                    ListItem(Label(f"{session['session_id']}  [dim]{when}[/dim]"))
                )
            yield list_view

    def action_select(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        if list_view.index is None:
            return
        index = list_view.index
        if 0 <= index < len(self._sessions):
            self.dismiss(self._sessions[index]["path"])

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ModelSelector", "SessionPicker"]
