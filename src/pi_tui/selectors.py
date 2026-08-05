"""选择器组件（3.6）：ModelSelector / SessionPicker。"""

from __future__ import annotations

from typing import Any, cast

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Input, Label, ListItem, ListView
from textual.widget import Widget
from datetime import datetime

from .lists import SelectList


def format_label_timestamp(iso_value: str | None) -> str:
    """ISO 时间戳 → 本地 HH:MM:SS（解析失败返回原文）。"""
    if not iso_value:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%H:%M:%S")
    except ValueError:
        return iso_value


class OverlayDialog(Widget):
    """overlay 化对话框基类：dismiss 通过 PiTuiApp 桥接关闭并回调。"""

    def dismiss(self, value: Any = None) -> None:
        close = getattr(self.app, "_close_overlay_dialog", None)
        if close is not None:
            close(self, value)


class CopyRequested(Message):
    """列表弹层请求复制选中项文本（按 c 触发，事件冒泡到宿主）。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class CopySelectedMixin(Widget):
    """列表弹层复制能力：选中行时直接复制完整内容到剪贴板。

    Textual 不支持鼠标选择，因此“选中即复制”：子类在各自的
    on_*_selected 中调用 copy_selected()，宿主处理 CopyRequested 消息
    写入剪贴板。
    """

    def copy_selected(self) -> None:
        text = self._selected_copy_text()
        if text:
            self.post_message(CopyRequested(text))

    def _selected_copy_text(self) -> str:
        return ""


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


class ModelSelector(OverlayDialog):
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


class SessionPicker(CopySelectedMixin, OverlayDialog):
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
            self.copy_selected()
            self.dismiss(self._sessions[index]["path"])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _selected_copy_text(self) -> str:
        list_view = self.query_one("#session-list", ListView)
        index = list_view.index
        if index is None or not (0 <= index < len(self._sessions)):
            return ""
        return str(self._sessions[index].get("path", ""))


def _flatten_tree(
    nodes: list[Any],
    leaf_id: str | None,
    depth: int = 0,
    show_label_timestamps: bool = False,
) -> list[tuple[int, str, str, str, Any]]:
    """把会话树展平为 [(depth, connector, label, node_id, node)]，供 TreeSelector 渲染。"""
    rows: list[tuple[int, str, str, str, Any]] = []
    for index, node in enumerate(nodes):
        is_last = index == len(nodes) - 1
        connector = "" if depth == 0 else ("└─" if is_last else "├─")
        marker = ">" if node.id == leaf_id else " "
        entry_type = node.entry.get("type", "?") if node.entry is not None else "?"
        label = ""
        if node.label:
            label = f" [{node.label}]"
            if show_label_timestamps and node.label_timestamp:
                label += f" @{format_label_timestamp(node.label_timestamp)}"
        rows.append(
            (depth, connector, f"{marker} {node.id[:8]} {entry_type}{label}", node.id, node)
        )
        rows.extend(_flatten_tree(node.children, leaf_id, depth + 1, show_label_timestamps))
    return rows


def _node_copy_text(node: Any) -> str:
    """取树节点可复制的完整文本：message 条目返回全文，其余返回 label。"""
    entry = node.entry or {}
    if entry.get("type") == "message":
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""
    return str(node.label or "")


TREE_FILTER_MODES = ("default", "no-tools", "user-only", "labeled-only", "all")

_SETTINGS_ENTRY_TYPES = {
    "label",
    "custom",
    "custom_message",
    "model_change",
    "thinking_level_change",
    "session_info",
}


def _entry_role(node: Any) -> str:
    entry = node.entry or {}
    if entry.get("type") == "message":
        return str(entry.get("role", ""))
    return str(entry.get("type", ""))


def node_passes_tree_filter(node: Any, mode: str) -> bool:
    """树过滤判定（对齐 TS tree-selector：default/no-tools/user-only/labeled-only/all）。"""
    entry_type = (node.entry or {}).get("type", "")
    role = _entry_role(node)
    if mode == "user-only":
        return role == "user"
    if mode == "no-tools":
        return not (entry_type in _SETTINGS_ENTRY_TYPES or role == "toolResult")
    if mode == "labeled-only":
        return node.label is not None
    if mode == "all":
        return True
    # default：隐藏设置/记账类条目。
    return entry_type not in _SETTINGS_ENTRY_TYPES


class TreeSelector(CopySelectedMixin, OverlayDialog):
    """会话树选择器（对齐 TS TreeSelectorComponent）：ASCII 树 + 键盘导航。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("f", "cycle_filter", "Cycle tree filter"),
        Binding("t", "toggle_label_timestamps", "Toggle label timestamps"),
    ]

    def __init__(
        self,
        tree: list[Any],
        leaf_id: str | None = None,
        filter_mode: str = "default",
        show_label_timestamps: bool = False,
    ) -> None:
        super().__init__()
        self._filter_mode = filter_mode if filter_mode in TREE_FILTER_MODES else "default"
        self._show_label_timestamps = show_label_timestamps
        self._tree = tree
        self._leaf_id = leaf_id
        self._rebuild_rows()
        self._rows: list[tuple[int, str, str, str, Any]] = []
        self._node_ids: list[str] = []
        self._apply_filter()

    @property
    def filter_mode(self) -> str:
        return self._filter_mode

    @property
    def show_label_timestamps(self) -> bool:
        return self._show_label_timestamps

    def _rebuild_rows(self) -> None:
        self._all_rows = _flatten_tree(
            self._tree,
            self._leaf_id,
            show_label_timestamps=self._show_label_timestamps,
        )

    def _title(self) -> str:
        suffix = {
            "no-tools": " [no-tools]",
            "user-only": " [user]",
            "labeled-only": " [labeled]",
            "all": " [all]",
        }.get(self._filter_mode, "")
        if self._show_label_timestamps:
            suffix += " [+label time]"
        # 转义方括号，避免被 Textual 当成 Rich markup 样式标签吞掉。
        suffix = suffix.replace("[", r"\[").replace("]", r"\]")
        return f"Session tree{suffix} (Enter: navigate, Esc: close, f: filter, t: label time)"

    def _apply_filter(self) -> None:
        self._rows = [
            row for row in self._all_rows if node_passes_tree_filter(row[4], self._filter_mode)
        ]
        # 节点 id 可能以数字开头（Textual id 非法），选择逻辑用索引反查。
        self._node_ids = [row[3] for row in self._rows]

    def _selected_copy_text(self) -> str:
        list_view = self.query_one("#tree-list", ListView)
        index = list_view.index
        if index is None or not (0 <= index < len(self._rows)):
            return ""
        _depth, _connector, _label, _node_id, node = self._rows[index]
        return _node_copy_text(node)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title(), classes="selector-title")
            yield ListView(id="tree-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#tree-list", ListView)
        for depth, connector, label, _node_id, _node in self._rows:
            indent = "  " * depth
            prefix = f"{indent}{connector} " if connector else indent
            list_view.append(ListItem(Label(f"{prefix}{label}")))
        if len(list_view.children) > 0:
            list_view.index = 0

    async def action_cycle_filter(self) -> None:
        """f：循环 default → no-tools → user-only → labeled-only → all。"""
        modes = TREE_FILTER_MODES
        self._filter_mode = modes[(modes.index(self._filter_mode) + 1) % len(modes)]
        self._apply_filter()
        self.query_one(Label).update(self._title())
        list_view = self.query_one("#tree-list", ListView)
        await list_view.clear()
        for depth, connector, label, _node_id, _node in self._rows:
            indent = "  " * depth
            prefix = f"{indent}{connector} " if connector else indent
            list_view.append(ListItem(Label(f"{prefix}{label}")))
        if len(list_view.children) > 0:
            list_view.index = 0

    async def action_toggle_label_timestamps(self) -> None:
        """t：切换 label 时间戳显示。"""
        self._show_label_timestamps = not self._show_label_timestamps
        self._rebuild_rows()
        self._apply_filter()
        self.query_one(Label).update(self._title())
        list_view = self.query_one("#tree-list", ListView)
        await list_view.clear()
        for depth, connector, label, _node_id, _node in self._rows:
            indent = "  " * depth
            prefix = f"{indent}{connector} " if connector else indent
            list_view.append(ListItem(Label(f"{prefix}{label}")))
        if len(list_view.children) > 0:
            list_view.index = 0

    def on_list_view_selected(self, event: Any) -> None:
        list_view = self.query_one("#tree-list", ListView)
        index = list_view.index
        if index is not None and 0 <= index < len(self._node_ids):
            self.copy_selected()
            self.dismiss(self._node_ids[index])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextInputDialog(OverlayDialog):
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


class ChoiceSelector(CopySelectedMixin, OverlayDialog):
    """通用选项列表弹层（settings 菜单子项等）。"""

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

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="selector-title")
            yield SelectList(self._options, current=self._current, list_id="choice-list")

    def on_select_list_selected(self, event: SelectList.Selected) -> None:
        self.copy_selected()
        self.dismiss(event.item.value)

    def on_select_list_cancelled(self, event: SelectList.Cancelled) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        """兼容旧 API：按当前列表索引选择。"""
        select = self.query_one(SelectList)
        item = select.selected_item
        self.dismiss(item.value if item is not None else None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _selected_copy_text(self) -> str:
        select = self.query_one(SelectList)
        item = select.selected_item
        return str(item.value) if item is not None else ""


class SettingsSelector(OverlayDialog):
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
            cast(Any, self.app).push_screen(
                ChoiceSelector(
                    item.get("label", key),
                    list(item.get("choices", [])),
                    current_text,
                ),
                callback=lambda value: self._apply_value(key, value),
            )
            return
        current = self._value(key)
        cast(Any, self.app).push_screen(
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


class ThinkingSelector(OverlayDialog):
    """思考级别选择器（对齐 TS thinking-selector）。"""

    def __init__(
        self,
        levels: list[str],
        current: str | None = None,
    ) -> None:
        super().__init__()
        self._levels = list(levels)
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Thinking level", classes="selector-title")
            yield SelectList(self._levels, current=self._current, list_id="thinking-list")

    def on_select_list_selected(self, event: SelectList.Selected) -> None:
        self.dismiss(event.item.value)

    def on_select_list_cancelled(self, event: SelectList.Cancelled) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        """兼容旧 API：按当前列表索引选择。"""
        select = self.query_one(SelectList)
        item = select.selected_item
        self.dismiss(item.value if item is not None else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class OAuthSelector(OverlayDialog):
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


class ScopedModelsSelector(OverlayDialog):
    """模型范围选择器：Enter 切换选中，Esc 保存（对齐 TS scoped-models-selector）。"""

    BINDINGS = [
        Binding("enter", "toggle_scoped", "Toggle"),
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
        self.action_toggle_scoped()

    def action_toggle_scoped(self) -> None:
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


class ExtensionSelector(OverlayDialog):
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


class TrustSelector(OverlayDialog):
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
            saved_decision = self._saved_decision
            check = (
                " ✓"
                if option.get("savedPath") == self._saved_decision
                and (
                    option.get("trusted") == saved_decision.get("decision")
                    if saved_decision is not None
                    else False
                )
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
