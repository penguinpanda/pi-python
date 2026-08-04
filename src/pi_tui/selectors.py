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
        current_key = f"{self._current.provider}/{self._current.id}" if self._current else None
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
            when = datetime.fromtimestamp(session["modified"]).strftime("%Y-%m-%d %H:%M")
            list_view.append(ListItem(Label(f"{session['session_id']}  [dim]{when}[/dim]")))
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
        rows.append((depth, connector, f"{marker} {node.id[:8]} {entry_type}{label}", node.id))
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
        for depth, connector, label, _node_id in self._rows:
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

    def __init__(self, message: str, placeholder: str = "", value: str = "") -> None:
        super().__init__()
        self._message = message
        self._placeholder = placeholder
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._message, classes="selector-title")
            yield Input(
                placeholder=self._placeholder,
                value=self._value,
                classes="selector-input",
            )

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceSelector(ModalScreen):
    """通用选项列表弹层（settings 菜单子项等）。"""

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        options: list[str],
        current: str | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = list(options)
        self._current = current
        self._selected = 0
        if current is not None and current in self._options:
            self._selected = self._options.index(current)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="selector-title")
            yield ListView(id="choice-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#choice-list", ListView)
        for index, option in enumerate(self._options):
            marker = ">" if index == self._selected else " "
            check = " ✓" if option == self._current else ""
            list_view.append(ListItem(Label(f"{marker} {option}{check}")))
        if len(list_view.children) > 0:
            list_view.index = self._selected
        list_view.focus()

    def on_list_view_selected(self, event: Any) -> None:
        self.action_select()

    def action_select(self) -> None:
        list_view = self.query_one("#choice-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._options):
            self.dismiss(self._options[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingsSelector(ModalScreen):
    """设置菜单（对齐 TS SettingsSelectorComponent 的 Python 子集）。

    items: [{"key", "label", "type": "bool"|"choice"|"string", "choices"?}]
    current: 当前合并后的 settings 字典。
    on_change(key, value)：持久化并应用。
    """

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        items: list[dict[str, Any]],
        current: dict[str, Any],
        on_change,
    ) -> None:
        super().__init__()
        self._items = list(items)
        self._current = dict(current)
        self._on_change = on_change

    def _value(self, key: str) -> Any:
        return self._current.get(key)

    def _label(self, item: dict[str, Any]) -> str:
        key = item["key"]
        value = self._value(key)
        if item.get("type") == "bool":
            display = "true" if value else "false"
        else:
            display = str(value) if value is not None else "(unset)"
        return f"{item['label']}: {display}"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Settings", classes="selector-title")
            yield ListView(id="settings-list")

    def on_mount(self) -> None:
        self._rebuild()
        self.query_one("#settings-list").focus()

    def _rebuild(self) -> None:
        list_view = self.query_one("#settings-list", ListView)
        list_view.clear()
        for item in self._items:
            list_view.append(ListItem(Label(self._label(item))))
        if len(list_view.children) > 0 and list_view.index is None:
            list_view.index = 0

    def on_list_view_selected(self, event: Any) -> None:
        self._select_item()

    def action_select(self) -> None:
        self._select_item()

    def _select_item(self) -> None:
        list_view = self.query_one("#settings-list", ListView)
        index = list_view.index
        if index is None or not (0 <= index < len(self._items)):
            self.dismiss(None)
            return
        item = self._items[index]
        item_type = item.get("type", "string")
        key = item["key"]
        if item_type == "bool":
            new_value = not bool(self._value(key))
            self._on_change(key, new_value)
            self._current[key] = new_value
            self._rebuild()
            return
        if item_type == "choice":
            current = self._value(key)
            current_text = str(current) if current is not None else None
            self.push_screen(
                ChoiceSelector(
                    item.get("label", key),
                    list(item.get("choices", [])),
                    current_text,
                ),
                callback=lambda value: self._apply_value(key, value),
            )
            return
        current = self._value(key)
        self.push_screen(
            TextInputDialog(
                f"{item.get('label', key)}:",
                value=str(current) if current is not None else "",
            ),
            callback=lambda value: self._apply_value(key, value),
        )

    def _apply_value(self, key: str, value) -> None:
        if value is None or value == "":
            return
        self._on_change(key, value)
        self._current[key] = value
        self._rebuild()

    def action_cancel(self) -> None:
        self.dismiss(None)


class ThinkingSelector(ModalScreen):
    """思考级别选择器（对齐 TS thinking-selector）。"""

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        levels: list[str],
        current: str | None = None,
    ) -> None:
        super().__init__()
        self._levels = list(levels)
        self._current = current
        self._selected = 0
        if current is not None and current in self._levels:
            self._selected = self._levels.index(current)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Thinking level", classes="selector-title")
            yield ListView(id="thinking-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#thinking-list", ListView)
        for index, level in enumerate(self._levels):
            marker = ">" if index == self._selected else " "
            check = " ✓" if level == self._current else ""
            list_view.append(ListItem(Label(f"{marker} {level}{check}")))
        if len(list_view.children) > 0:
            list_view.index = self._selected
        list_view.focus()

    def on_list_view_selected(self, event: Any) -> None:
        self.action_select()

    def action_select(self) -> None:
        list_view = self.query_one("#thinking-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._levels):
            self.dismiss(self._levels[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class OAuthSelector(ModalScreen):
    """OAuth provider 选择器（登录/登出；对齐 TS oauth-selector）。"""

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        providers: list[tuple[str, str, bool]],
        *,
        mode: str = "login",
    ) -> None:
        super().__init__()
        self._providers = list(providers)
        self._mode = mode
        self._selected = 0

    def compose(self) -> ComposeResult:
        title = "Login provider" if self._mode == "login" else "Logout provider"
        with Vertical():
            yield Label(title, classes="selector-title")
            yield ListView(id="oauth-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#oauth-list", ListView)
        for index, (provider_id, name, logged_in) in enumerate(self._providers):
            marker = ">" if index == self._selected else " "
            status = "logged in" if logged_in else "not logged in"
            list_view.append(ListItem(Label(f"{marker} {name} ({provider_id}) [{status}]")))
        if len(list_view.children) > 0:
            list_view.index = 0
        list_view.focus()

    def on_list_view_selected(self, event: Any) -> None:
        self.action_select()

    def action_select(self) -> None:
        list_view = self.query_one("#oauth-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._providers):
            self.dismiss(self._providers[index][0])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ScopedModelsSelector(ModalScreen):
    """模型范围选择器：Enter 切换选中，Esc 保存（对齐 TS scoped-models-selector）。"""

    BINDINGS = [
        Binding("enter", "toggle", "Toggle"),
        Binding("escape", "cancel", "Save and close"),
    ]

    def __init__(
        self,
        models: list[Any],
        selected: set[tuple[str, str]] | None = None,
        current: Any | None = None,
    ) -> None:
        super().__init__()
        self._models = list(models)
        self._selected = set(selected or {})
        self._current = current

    def _key(self, model) -> tuple[str, str]:
        return (model.provider, model.id)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Scoped models (Enter: toggle, Esc: save)", classes="selector-title")
            yield ListView(id="scoped-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#scoped-list", ListView)
        for model in self._models:
            key = self._key(model)
            check = " ✓" if key in self._selected else ""
            marker = ">" if self._current is not None and key == self._key(self._current) else " "
            list_view.append(ListItem(Label(f"{marker} {model.provider}/{model.id}{check}")))
        if len(list_view.children) > 0:
            list_view.index = 0
        list_view.focus()

    def _rebuild(self) -> None:
        list_view = self.query_one("#scoped-list", ListView)
        list_view.clear()
        for model in self._models:
            key = self._key(model)
            check = " ✓" if key in self._selected else ""
            list_view.append(ListItem(Label(f"  {model.provider}/{model.id}{check}")))
        if len(list_view.children) > 0:
            list_view.index = 0

    def on_list_view_selected(self, event: Any) -> None:
        self.action_toggle()

    def action_toggle(self) -> None:
        list_view = self.query_one("#scoped-list", ListView)
        index = list_view.index
        if index is None or not (0 <= index < len(self._models)):
            return
        key = self._key(self._models[index])
        if key in self._selected:
            self._selected.discard(key)
        else:
            self._selected.add(key)
        self._rebuild()

    def action_cancel(self) -> None:
        self.dismiss(self._selected)


class ExtensionSelector(ModalScreen):
    """扩展列表选择器（对齐 TS extension-selector）。"""

    BINDINGS = [
        Binding("enter", "select", "Show details"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, extensions: list[dict[str, Any]]) -> None:
        super().__init__()
        self._extensions = list(extensions)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Extensions", classes="selector-title")
            yield ListView(id="extension-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#extension-list", ListView)
        for index, extension in enumerate(self._extensions):
            marker = ">" if index == 0 else " "
            label = extension.get("label", extension.get("path", "?"))
            list_view.append(ListItem(Label(f"{marker} {label}")))
        if len(list_view.children) > 0:
            list_view.index = 0
        list_view.focus()

    def on_list_view_selected(self, event: Any) -> None:
        self.action_select()

    def action_select(self) -> None:
        list_view = self.query_one("#extension-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._extensions):
            self.dismiss(self._extensions[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TrustSelector(ModalScreen):
    """项目信任选择器（对齐 TS TrustSelectorComponent）。"""

    BINDINGS = [
        Binding("enter", "select", "Select"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        cwd: str,
        saved_decision: dict | None = None,
        project_trusted: bool = False,
    ) -> None:
        super().__init__()
        from pi_coding_agent.trust import get_project_trust_options

        self._cwd = cwd
        self._saved_decision = saved_decision
        self._project_trusted = project_trusted
        self._options = get_project_trust_options(cwd)
        # 预选当前已保存的选项。
        self._selected = 0
        if saved_decision is not None:
            for index, option in enumerate(self._options):
                if option.get("savedPath") == saved_decision.get("path") and option.get(
                    "trusted"
                ) == saved_decision.get("decision"):
                    self._selected = index
                    break

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Project trust", classes="selector-title")
            yield Label(self._cwd, classes="selector-title")
            status = self._format_decision()
            yield Label(
                f"Saved decision: {status}  |  Current session: "
                f"{'trusted' if self._project_trusted else 'untrusted'}",
                classes="selector-title",
            )
            yield ListView(id="trust-list")

    def _format_decision(self) -> str:
        entry = self._saved_decision
        if entry is None:
            return "none"
        label = "trusted" if entry.get("decision") else "untrusted"
        if entry.get("path") != self._cwd:
            return f"{label} (inherited from {entry.get('path')})"
        return f"{label} ({entry.get('path')})"

    def on_mount(self) -> None:
        list_view = self.query_one("#trust-list", ListView)
        for index, option in enumerate(self._options):
            marker = ">" if index == self._selected else " "
            check = (
                " ✓"
                if option.get("savedPath") == self._saved_decision
                and (option.get("trusted") == self._saved_decision.get("decision"))
                else ""
            )
            list_view.append(ListItem(Label(f"{marker} {option['label']}{check}")))
        if len(list_view.children) > 0:
            list_view.index = self._selected
        list_view.focus()

    def on_list_view_selected(self, event: Any) -> None:
        self.action_select()

    def action_select(self) -> None:
        list_view = self.query_one("#trust-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._options):
            self.dismiss(self._options[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = [
    "ModelSelector",
    "SessionPicker",
    "TreeSelector",
    "TextInputDialog",
    "ChoiceSelector",
    "SettingsSelector",
    "ThinkingSelector",
    "OAuthSelector",
    "ScopedModelsSelector",
    "ExtensionSelector",
    "TrustSelector",
]
