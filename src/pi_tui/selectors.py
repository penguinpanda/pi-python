"""选择器组件（3.6）：ModelSelector / SessionPicker。"""

from __future__ import annotations

from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView


def _model_label(model) -> str:
    return f"{model.provider}/{model.id}  [dim]{model.name}[/dim]"


class _SearchInput(Input):
    """搜索输入框：把 ↑↓/Enter/Esc 转发给选择器，避免按键被输入框吞掉。"""

    def __init__(self, owner: "ModelSelector", **kwargs) -> None:
        super().__init__(**kwargs)
        self._owner = owner

    async def _on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down", "enter", "escape"):
            event.stop()
            event.prevent_default()
            if event.key == "up":
                self._owner._move(-1)
            elif event.key == "down":
                self._owner._move(1)
            elif event.key == "enter":
                self._owner.action_select()
            else:
                self._owner.action_cancel()
            return
        await super()._on_key(event)


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
            yield _SearchInput(
                self,
                placeholder="Search models...",
                value=self._query,
                classes="selector-input",
            )
            yield ListView(id="model-list")

    def _move(self, direction: int) -> None:
        """在模型列表中移动选择（供搜索框按键转发）。"""
        list_view = self.query_one("#model-list", ListView)
        count = len(list_view.children)
        if count == 0:
            return
        current = list_view.index
        index = 0 if current is None else current
        list_view.index = (index + direction) % count

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
            # 不设置 id：模型 id 可能含冒号等 Textual 非法字符
            # （如 ollama/qwen3:30b），选择逻辑只依赖列表索引。
            list_view.append(ListItem(Label(f"{marker} {_model_label(model)}")))
        # 初始选中第一项，否则不按方向键直接 Enter 不会有 Selected 事件。
        if len(list_view.children) > 0 and list_view.index is None:
            list_view.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._query = event.value
        self._rebuild()

    def action_select(self) -> None:
        list_view = self.query_one("#model-list", ListView)
        index = list_view.index
        if index is None:
            index = 0
        filtered = self._filtered()
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
            yield ListView(id="session-list")

    def on_mount(self) -> None:
        """挂载后再填充列表（compose 阶段 ListView 尚未挂载，append 会抛 MountError）。"""
        from datetime import datetime

        list_view = self.query_one("#session-list", ListView)
        for session in self._sessions:
            when = datetime.fromtimestamp(session["modified"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            list_view.append(
                ListItem(Label(f"{session['session_id']}  [dim]{when}[/dim]"))
            )
        if len(list_view.children) > 0:
            list_view.index = 0

    def action_select(self) -> None:
        """兜底选择（ListView 未消费 Enter 时）。"""
        list_view = self.query_one("#session-list", ListView)
        index = list_view.index
        if index is None:
            index = 0
        if 0 <= index < len(self._sessions):
            self.dismiss(self._sessions[index]["path"])

    def on_list_view_selected(self, event: Any) -> None:
        """ListView 的 Enter 选择（Textual 会先于屏幕绑定消费 enter）。"""
        list_view = self.query_one("#session-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._sessions):
            self.dismiss(self._sessions[index]["path"])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


def _flatten_tree(
    nodes: list[Any],
    leaf_id: str | None,
    depth: int = 0,
) -> list[tuple[int, str, str, str]]:
    """把会话树展平为 [(depth, connector, label, node_id)]，供 TreeSelector 渲染。"""
    rows: list[tuple[int, str, str, str]] = []
    for index, node in enumerate(nodes):
        is_last = index == len(nodes) - 1
        connector = "" if depth == 0 else ("└─" if is_last else "├─")
        marker = ">" if node.id == leaf_id else " "
        entry_type = node.entry.get("type", "?") if node.entry is not None else "?"
        label = f" [{node.label}]" if node.label else ""
        rows.append(
            (depth, connector, f"{marker} {node.id[:8]} {entry_type}{label}", node.id)
        )
        rows.extend(_flatten_tree(node.children, leaf_id, depth + 1))
    return rows


class TreeSelector(ModalScreen):
    """会话树选择器（对齐 TS TreeSelectorComponent）：ASCII 树 + 键盘导航。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, tree: list[Any], leaf_id: str | None = None) -> None:
        super().__init__()
        self._rows = _flatten_tree(tree, leaf_id)
        # 节点 id 可能以数字开头（Textual id 非法），选择逻辑用索引反查。
        self._node_ids = [row[3] for row in self._rows]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Session tree (Enter: navigate, Esc: close)", classes="selector-title")
            yield ListView(id="tree-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#tree-list", ListView)
        for depth, connector, label, node_id in self._rows:
            indent = "  " * depth
            prefix = f"{indent}{connector} " if connector else indent
            list_view.append(ListItem(Label(f"{prefix}{label}")))
        if len(list_view.children) > 0:
            list_view.index = 0

    def on_list_view_selected(self, event: Any) -> None:
        list_view = self.query_one("#tree-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._node_ids):
            self.dismiss(self._node_ids[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextInputDialog(ModalScreen):
    """通用文本输入弹层（TUI 内 OAuth 登录等需要用户输入的场景）。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, message: str, placeholder: str = "") -> None:
        super().__init__()
        self._message = message
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._message, classes="selector-title")
            yield Input(placeholder=self._placeholder, classes="selector-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ModelSelector", "SessionPicker", "TreeSelector", "TextInputDialog"]
