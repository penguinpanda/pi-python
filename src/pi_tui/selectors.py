"""选择器组件（引擎版）：Model / Session / Tree / OAuth / Scoped / Extension / Trust 等。"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import (
    Input,
    Label,
    Message,
    SelectItem,
    SelectList,
    Vertical,
    Widget,
)
from pi_tui.keybindings import DEFAULT_SESSION_PICKER_KEYBINDINGS


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

    def handle_key(self, key: Key) -> bool:
        return False


class CopyRequested(Message):
    """列表弹层请求复制选中项文本（按 c 触发，事件冒泡到宿主）。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class CopySelectedMixin:
    """列表弹层复制能力：选中行时直接复制完整内容到剪贴板。"""

    def copy_selected(self) -> None:
        text = self._selected_copy_text()
        if text and isinstance(self, Widget):
            self.post_message(CopyRequested(text), "")

    def _selected_copy_text(self) -> str:
        return ""


def _model_label(model) -> str:
    """对齐 TS model-selector：`model-id [provider]`（id 在前、provider badge 在后）。"""
    return f"{model.id} [{model.provider}]"


class _ResultSelectList(SelectList):
    """选择结果直接回调（不经 App 消息总线）。"""

    def __init__(
        self,
        items,
        *,
        on_selected: Callable[[SelectItem | None], None],
        on_cancelled: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(items, **kwargs)
        self._on_selected = on_selected
        self._on_cancelled = on_cancelled

    def handle_key(self, key: Key) -> bool:
        if key.name == "enter":
            self._on_selected(self.selected_item)
            return True
        if key.name == "escape":
            self._on_cancelled()
            return True
        return super().handle_key(key)


class ModelSelector(OverlayDialog):
    """模型选择器：分组显示 + 实时搜索 + 键盘导航（对齐 TS ModelSelectorComponent）。"""

    def __init__(self, models: list[Any], current: Any | None = None) -> None:
        super().__init__()
        self._models = list(models)
        self._current = current
        current_key = f"{current.provider}/{current.id}" if current is not None else None
        items = [
            SelectItem(
                value=f"{model.provider}/{model.id}",
                label=_model_label(model),
            )
            for model in self._models
        ]
        self._list = _ResultSelectList(
            items,
            current=current_key,
            enable_search=True,
            search_placeholder="Search models...",
            max_height=11,
            show_scroll_indicator=True,
            on_selected=self._on_selected,
            on_cancelled=lambda: self.dismiss(None),
        )
        self._body = Vertical()
        self._body.mount(Label("Select model", height=1))
        self._body.mount(self._list)

    def _on_selected(self, item: SelectItem | None) -> None:
        if item is None:
            self.dismiss(None)
            return
        for model in self._models:
            if f"{model.provider}/{model.id}" == item.value:
                self.dismiss(model)
                return
        self.dismiss(None)

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


class SessionPicker(CopySelectedMixin, OverlayDialog):
    """会话恢复选择器：排序 / 过滤 / scope / 路径 / 删除 / 重命名。

    `model` 由调用方（pi_coding_agent）构造的 SessionPickerModel 实例注入，
    保持 pi_tui 不反向依赖 pi_coding_agent（对齐 TS packages/tui 分层边界）。
    """

    def __init__(
        self,
        model: Any,
        *,
        on_rename: Callable[[str, str], Any] | None = None,
        on_delete: Callable[[str], Any] | None = None,
        reload_sessions: Callable[[], Any] | None = None,
        keybindings_manager=None,
        keybindings: dict[str, str] | None = None,
        max_height: int = 12,
    ) -> None:
        super().__init__()
        self._model = model
        self._on_rename = on_rename
        self._on_delete = on_delete
        self._reload_sessions = reload_sessions
        self._keybindings_manager = keybindings_manager
        self._session_keys = {
            action: binding.key for action, binding in DEFAULT_SESSION_PICKER_KEYBINDINGS.items()
        }
        if keybindings:
            self._session_keys.update(keybindings)
        self._max_height = max(4, int(max_height))
        self._confirming_path: str | None = None
        self._status: tuple[str, str] | None = None
        self._rename_mode = False
        self._rename_target: str | None = None
        self._rename_input = Input()
        self._query = ""

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------

    def _selected_copy_text(self) -> str:
        path = self._model.selected_path
        return path or ""

    def _select_current(self) -> None:
        path = self._model.selected_path
        if path is None:
            self.dismiss(None)
            return
        self.copy_selected()
        self.dismiss(path)

    # ------------------------------------------------------------------
    # 删除 / 重命名
    # ------------------------------------------------------------------

    def _start_delete_confirmation(self) -> None:
        if self._model.is_selected_current:
            self._set_status("error", "Cannot delete the currently active session")
            return
        self._confirming_path = self._model.selected_path
        self.refresh()

    async def _confirm_delete(self) -> None:
        path = self._confirming_path
        self._confirming_path = None
        if path is None:
            return
        if self._on_delete is None:
            self._set_status("error", "Deleting sessions is not available")
            return
        try:
            result = self._on_delete(path)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str) and result:
                self._set_status("error", result)
                return
        except Exception as exc:
            self._set_status("error", f"Failed to delete: {exc}")
            return
        self._model.remove_path(path)
        await self._refresh_after_mutation()
        self._set_status("info", "Session deleted")
        self.refresh()

    def _enter_rename_mode(self) -> None:
        session = self._model.selected_node
        if session is None:
            return
        self._rename_mode = True
        self._rename_target = session.session.path
        self._rename_input.value = session.session.name or ""
        self._rename_input.cursor = len(self._rename_input.value)
        self.refresh()

    def _exit_rename_mode(self) -> None:
        self._rename_mode = False
        self._rename_target = None
        self.refresh()

    async def _confirm_rename(self) -> None:
        target = self._rename_target
        value = self._rename_input.value
        self._exit_rename_mode()
        if target is None or not value.strip():
            return
        if self._on_rename is None:
            self._set_status("error", "Renaming sessions is not available")
            return
        try:
            result = self._on_rename(target, value)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str) and result:
                self._set_status("error", result)
                return
        except Exception as exc:
            self._set_status("error", f"Failed to rename: {exc}")
            return
        await self._refresh_after_mutation()
        self._set_status("info", "Session renamed")
        self.refresh()

    async def _refresh_after_mutation(self) -> None:
        if self._reload_sessions is None:
            return
        try:
            result = self._reload_sessions()
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, tuple) and len(result) == 2:
                current, all_sessions = result
                self._model.current_sessions = list(current)
                if all_sessions is not None:
                    self._model.all_sessions = list(all_sessions)
                self._model._refresh()
        except Exception:
            self._model._refresh()

    def _set_status(self, kind: str, message: str) -> None:
        self._status = (kind, message)

    # ------------------------------------------------------------------
    # 键盘
    # ------------------------------------------------------------------

    def handle_key(self, key: Key) -> bool:
        if self._rename_mode:
            return self._handle_rename_key(key)
        if self._confirming_path is not None:
            if key.name == "enter":
                self._run(self._confirm_delete())
            elif key.name == "escape":
                self._confirming_path = None
                self.refresh()
            return True

        name = key.name
        if name == self._session_key("app.session.toggleScope"):
            self._model.toggle_scope()
            self.refresh()
            return True
        if name == self._session_key("app.session.toggleSort"):
            self._model.toggle_sort()
            self.refresh()
            return True
        if name == self._session_key("app.session.toggleNamedFilter"):
            self._model.toggle_name_filter()
            self.refresh()
            return True
        if name == self._session_key("app.session.togglePath"):
            self._model.toggle_path()
            self.refresh()
            return True
        if name == self._session_key("app.session.delete"):
            self._start_delete_confirmation()
            return True
        if name == self._session_key("app.session.rename"):
            self._enter_rename_mode()
            return True
        if name == self._session_key("app.session.deleteNoninvasive"):
            if self._query:
                self._query = self._query[:-1]
                self._model.set_query(self._query)
            else:
                self._start_delete_confirmation()
            self.refresh()
            return True
        if name == "up":
            self._model.move_selection(-1)
            self.refresh()
            return True
        if name == "down":
            self._model.move_selection(1)
            self.refresh()
            return True
        if name == "pageup":
            self._model.page_selection(-(self._max_height - 3))
            self.refresh()
            return True
        if name == "pagedown":
            self._model.page_selection(self._max_height - 3)
            self.refresh()
            return True
        if name == "enter":
            self._select_current()
            return True
        if name == "escape":
            self.dismiss(None)
            return True
        if name == "backspace":
            if self._query:
                self._query = self._query[:-1]
                self._model.set_query(self._query)
            self.refresh()
            return True
        if key.char is not None and key.char.isprintable():
            self._query += key.char
            self._model.set_query(self._query)
            self.refresh()
            return True
        return False

    def _handle_rename_key(self, key: Key) -> bool:
        if key.name == "escape":
            self._exit_rename_mode()
            return True
        if key.name == "enter":
            self._run(self._confirm_rename())
            return True
        if self._rename_input.handle_key(key):
            self.refresh()
            return True
        return False

    def _run(self, coro) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._async_task = asyncio.create_task(coro)

    def _session_key(self, action: str) -> str:
        if self._keybindings_manager is not None:
            getter = getattr(self._keybindings_manager, "get_session_picker_key", None)
            if getter is not None:
                value = getter(action)
                if value:
                    return value
        return self._session_keys.get(action, "")

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render(self, width: int, height: int) -> list[Line]:
        if self._rename_mode:
            return self._render_rename(width, height)
        lines: list[Line] = []
        lines.append(line_from_text(self._title(), width))
        lines.append(line_from_text(self._query or "Search sessions...", width, _dim_style()))
        if self._status is not None:
            lines.append(line_from_text(self._status[1], width))
        elif self._confirming_path is not None:
            lines.append(
                line_from_text(
                    "Delete session? Enter to confirm · Esc to cancel",
                    width,
                )
            )
        else:
            lines.append(line_from_text(self._hints(), width))

        visible = min(height - len(lines), len(self._model.rows))
        start, end = _visible_window(
            self._model.selected_index,
            visible,
            len(self._model.rows),
        )
        for index in range(start, end):
            node = self._model.rows[index]
            selected = index == self._model.selected_index
            style = None
            if selected:
                from rich.style import Style

                style = Style(reverse=True)
            prefix = self._tree_prefix(node)
            text = self._row_text(node.session, width - len(prefix))
            lines.append(line_from_text(f"{prefix}{text}", width, style))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines

    def _render_rename(self, width: int, height: int) -> list[Line]:
        lines = [
            line_from_text("Rename Session", width),
            line_from_text("", width),
        ]
        lines.extend(self._rename_input.render(width, max(1, height - 4)))
        lines.append(line_from_text("Enter to save · Esc to cancel", width, _dim_style()))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines

    def _title(self) -> str:
        scope = "Current Folder" if self._model.scope.value == "current" else "All"
        sort = {
            "threaded": "Threaded",
            "recent": "Recent",
            "relevance": "Fuzzy",
        }[self._model.sort_mode.value]
        name = "Named" if self._model.name_filter.value == "named" else "All"
        path = "on" if self._model.show_path else "off"
        return f"Resume Session ({scope})  Name: {name}  Sort: {sort}  Path: {path}"

    def _hints(self) -> str:
        parts = [
            "Tab: scope",
            "Ctrl+S: sort",
            "Ctrl+N: named",
            "Ctrl+P: path",
            "Ctrl+D: delete",
            "Ctrl+R: rename",
            're:<regex> · "phrase"',
        ]
        return " · ".join(parts)

    def _row_text(self, session, width: int) -> str:
        text = session.name or session.first_message or session.session_id
        right = f"{session.message_count} {_age_text(session.modified)}"
        if self._model.scope.value == "all" and session.cwd:
            right = f"{_shorten_path(session.cwd)} {right}"
        if self._model.show_path:
            right = f"{_shorten_path(session.path)} {right}"
        available = max(10, width - len(right) - 3)
        text = _truncate(text, available)
        return f" {text}  {right}"

    def _tree_prefix(self, node) -> str:
        if node.depth == 0:
            return ""
        parts = ["│  " if continues else "   " for continues in node.ancestor_continues]
        branch = "└─ " if node.is_last else "├─ "
        return "".join(parts) + branch

    def content_size(self) -> tuple[int, int]:
        return (1000, min(3 + len(self._model.rows), self._max_height))


def _dim_style():
    from rich.style import Style

    return Style(dim=True)


def _visible_window(selected: int, visible: int, total: int) -> tuple[int, int]:
    if total <= visible or visible <= 0:
        return (0, total)
    start = max(0, min(selected - visible // 2, total - visible))
    return (start, min(start + visible, total))


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _shorten_path(path: str) -> str:
    home = str(Path.home())
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def _age_text(timestamp: float) -> str:
    diff = datetime.now().timestamp() - timestamp
    if diff < 60:
        return "now"
    if diff < 3600:
        return f"{int(diff // 60)}m"
    if diff < 86400:
        return f"{int(diff // 3600)}h"
    if diff < 604800:
        return f"{int(diff // 86400)}d"
    if diff < 2592000:
        return f"{int(diff // 604800)}w"
    if diff < 31536000:
        return f"{int(diff // 2592000)}mo"
    return f"{int(diff // 31536000)}y"


def _flatten_tree(
    nodes: list[Any],
    leaf_id: str | None,
    depth: int = 0,
    show_label_timestamps: bool = False,
) -> list[tuple[int, str, str, str, Any]]:
    """把会话树展平为 [(depth, connector, label, node_id, node)]。"""
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
    """树过滤判定（对齐 TS tree-selector）。"""
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
    return entry_type not in _SETTINGS_ENTRY_TYPES


class TreeSelector(CopySelectedMixin, OverlayDialog):
    """会话树选择器（对齐 TS TreeSelectorComponent）：ASCII 树 + 键盘导航。"""

    def __init__(
        self,
        tree: list[Any],
        leaf_id: str | None = None,
        filter_mode: str = "default",
        show_label_timestamps: bool = False,
        on_label_edit=None,
    ) -> None:
        super().__init__()
        self._filter_mode = filter_mode if filter_mode in TREE_FILTER_MODES else "default"
        self._show_label_timestamps = show_label_timestamps
        self._tree = tree
        self._leaf_id = leaf_id
        self._selected_index = 0
        self._folded: set[str] = set()
        self._on_label_edit = on_label_edit
        self._all_rows = _flatten_tree(
            self._tree,
            self._leaf_id,
            show_label_timestamps=self._show_label_timestamps,
        )
        self._id_to_node: dict[str, Any] = {row[3]: row[4] for row in self._all_rows}
        self._rows: list[tuple[int, str, str, str, Any]] = []
        self._node_ids: list[str] = []
        self._apply_filter()

    @property
    def filter_mode(self) -> str:
        return self._filter_mode

    @property
    def show_label_timestamps(self) -> bool:
        return self._show_label_timestamps

    def _title(self) -> str:
        suffix = {
            "no-tools": " [no-tools]",
            "user-only": " [user]",
            "labeled-only": " [labeled]",
            "all": " [all]",
        }.get(self._filter_mode, "")
        if self._show_label_timestamps:
            suffix += " [+label time]"
        return f"Session tree{suffix} (Enter: navigate, Esc: close, f: filter, t: label time)"

    def _apply_filter(self) -> None:
        rows = [row for row in self._all_rows if node_passes_tree_filter(row[4], self._filter_mode)]
        if self._folded:
            # 折叠节点后代不显示（对齐 TS foldedNodes 过滤）。
            rows = [row for row in rows if not self._has_folded_ancestor(row[4])]
        self._rows = rows
        self._node_ids = [row[3] for row in self._rows]
        self._selected_index = 0

    def _has_folded_ancestor(self, node: Any) -> bool:
        """节点是否位于某个已折叠节点之下（parent_id 链）。"""
        current = node
        seen: set[str] = set()
        while current is not None:
            parent_id = getattr(current, "parent_id", None)
            if parent_id in self._folded:
                return True
            if parent_id is None or parent_id in seen:
                return False
            seen.add(parent_id)
            current = self._id_to_node.get(parent_id)
        return False

    def _is_foldable(self, node_id: str) -> bool:
        """节点是否有（未被过滤掉的）子行。"""
        depths = [row[0] for row in self._all_rows if row[3] == node_id]
        if not depths:
            return False
        depth = depths[0]
        for row in self._all_rows:
            if row[0] > depth and self._has_ancestor_in_tree(row[4], node_id):
                if node_passes_tree_filter(row[4], self._filter_mode):
                    return True
        return False

    def _has_ancestor_in_tree(self, node: Any, ancestor_id: str) -> bool:
        """node 是否在 ancestor_id 之下（不含自身）。"""
        current = node
        seen: set[str] = set()
        while current is not None:
            parent_id = getattr(current, "parent_id", None)
            if parent_id == ancestor_id:
                return True
            if parent_id is None or parent_id in seen:
                return False
            seen.add(parent_id)
            current = self._id_to_node.get(parent_id)
        return False

    def _fold_or_jump_up(self) -> None:
        if not self._rows:
            return
        current = self._node_ids[self._selected_index]
        if self._is_foldable(current) and current not in self._folded:
            self._folded.add(current)
            self._apply_filter()
            self.refresh()
            return
        # 无折叠：跳到分支段起点（向上最近的更浅行）。
        depth = self._rows[self._selected_index][0]
        for index in range(self._selected_index - 1, -1, -1):
            if self._rows[index][0] < depth:
                self._selected_index = index
                break
        self.refresh()

    def _unfold_or_jump_down(self) -> None:
        if not self._rows:
            return
        current = self._node_ids[self._selected_index]
        if current in self._folded:
            self._folded.discard(current)
            self._apply_filter()
            self.refresh()
            return
        # 无展开：跳到下一个分支段起点（向下最近的更浅行）。
        depth = self._rows[self._selected_index][0]
        for index in range(self._selected_index + 1, len(self._rows)):
            if self._rows[index][0] <= depth:
                self._selected_index = index
                break
        self.refresh()

    def _selected_copy_text(self) -> str:
        if 0 <= self._selected_index < len(self._rows):
            return _node_copy_text(self._rows[self._selected_index][4])
        return ""

    def handle_key(self, key: Key) -> bool:
        name = key.name
        if name == "up":
            if self._rows:
                self._selected_index = (self._selected_index - 1) % len(self._rows)
            self.refresh()
            return True
        if name == "down":
            if self._rows:
                self._selected_index = (self._selected_index + 1) % len(self._rows)
            self.refresh()
            return True
        if name in ("ctrl+left", "alt+left"):
            # 折叠或跳到分支段起点（对齐 TS app.tree.foldOrUp）。
            self._fold_or_jump_up()
            return True
        if name in ("ctrl+right", "alt+right"):
            # 展开或跳到下一个分支段起点（对齐 TS app.tree.unfoldOrDown）。
            self._unfold_or_jump_down()
            return True
        if name == "enter":
            if 0 <= self._selected_index < len(self._node_ids):
                self.copy_selected()
                self.dismiss(self._node_ids[self._selected_index])
            else:
                self.dismiss(None)
            return True
        if name == "escape":
            self.dismiss(None)
            return True
        if name == "f":
            modes = TREE_FILTER_MODES
            self._filter_mode = modes[(modes.index(self._filter_mode) + 1) % len(modes)]
            self._apply_filter()
            self.refresh()
            return True
        if name == "t":
            self._show_label_timestamps = not self._show_label_timestamps
            self._all_rows = _flatten_tree(
                self._tree,
                self._leaf_id,
                show_label_timestamps=self._show_label_timestamps,
            )
            self._apply_filter()
            self.refresh()
            return True
        if name == "shift+l":
            # 编辑当前节点 label（对齐 TS app.tree.editLabel）。
            if 0 <= self._selected_index < len(self._rows):
                row = self._rows[self._selected_index]
                if self._on_label_edit is not None:
                    self._on_label_edit(row[3], row[4].label)
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = [line_from_text(self._title(), width)]
        visible = min(height - 1, len(self._rows))
        for index in range(visible):
            depth, connector, label, node_id, _node = self._rows[index]
            if node_id in self._folded:
                label = f"⊞ {label}"
            elif self._is_foldable(node_id):
                label = f"⊟ {label}"
            indent = "  " * depth
            prefix = f"{indent}{connector} " if connector else indent
            text = f"{prefix}{label}"
            style = None
            if index == self._selected_index:
                from rich.style import Style

                style = Style(reverse=True)
            lines.append(line_from_text(text, width, style))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines

    def content_size(self) -> tuple[int, int]:
        return (1000, min(1 + len(self._rows), 20))


class TextInputDialog(OverlayDialog):
    """通用文本输入弹层（TUI 内 OAuth 登录等需要用户输入的场景）。"""

    def __init__(self, message: str, placeholder: str = "", value: str = "") -> None:
        super().__init__()
        self._message = message
        self._placeholder = placeholder
        self._value = value
        self._input = Input(value=value, placeholder=placeholder)
        self._body = Vertical()
        self._title_label = Label(self._message, height=1)
        self._body.mount(self._title_label)
        self._body.mount(self._input)

    def update_title(self, title: str) -> None:
        """更新标题（倒计时提示等动态文案）。"""
        self._message = title
        self._title_label.update(title)

    def handle_key(self, key: Key) -> bool:
        if self._input.handle_key(key):
            return True
        if key.name == "escape":
            self.dismiss(None)
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


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
        self._list = _ResultSelectList(
            self._options,
            current=current,
            enable_search=True,
            max_height=12,
            on_selected=self._on_selected,
            on_cancelled=lambda: self.dismiss(None),
        )
        self._body = Vertical()
        self._title_label = Label(title, height=1)
        self._body.mount(self._title_label)
        self._body.mount(self._list)

    def update_title(self, title: str) -> None:
        """更新标题（倒计时提示等动态文案）。"""
        self._title = title
        self._title_label.update(title)

    def _on_selected(self, item: SelectItem | None) -> None:
        if item is None:
            self.dismiss(None)
            return
        self.copy_selected()
        self.dismiss(item.value)

    def _selected_copy_text(self) -> str:
        item = self._list.selected_item
        return str(item.value) if item is not None else ""

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


class SettingsSelector(OverlayDialog):
    """设置菜单：bool 循环 / choice 弹选择器 / string 弹输入框。"""

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
        self._selected_index = 0

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

    def handle_key(self, key: Key) -> bool:
        name = key.name
        if name == "up":
            self._selected_index = (self._selected_index - 1) % len(self._items)
            self.refresh()
            return True
        if name == "down":
            self._selected_index = (self._selected_index + 1) % len(self._items)
            self.refresh()
            return True
        if name == "enter":
            self._select_item()
            return True
        if name == "escape":
            self.dismiss(None)
            return True
        return False

    def _select_item(self) -> None:
        if not (0 <= self._selected_index < len(self._items)):
            self.dismiss(None)
            return
        item = self._items[self._selected_index]
        item_type = item.get("type", "string")
        key = item["key"]
        if item_type == "bool":
            new_value = not bool(self._value(key))
            self._on_change(key, new_value)
            self._current[key] = new_value
            self.refresh()
            return
        if item_type == "choice":
            current = self._value(key)
            current_text = str(current) if current is not None else None
            cast_app = getattr(self.app, "push_screen", None)
            if cast_app is not None:
                cast_app(
                    ChoiceSelector(
                        item.get("label", key),
                        list(item.get("choices", [])),
                        current_text,
                    ),
                    callback=lambda value: self._apply_value(key, value),
                )
            return
        current = self._value(key)
        cast_app = getattr(self.app, "push_screen", None)
        if cast_app is not None:
            cast_app(
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
        self.refresh()

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = [line_from_text("Settings", width)]
        visible = min(height - 1, len(self._items))
        for index in range(visible):
            style = None
            if index == self._selected_index:
                from rich.style import Style

                style = Style(reverse=True)
            lines.append(line_from_text(self._label(self._items[index]), width, style))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines

    def content_size(self) -> tuple[int, int]:
        return (1000, min(1 + len(self._items), 16))


class ThinkingSelector(OverlayDialog):
    """思考级别选择器。"""

    def __init__(self, levels: list[str], current: str | None = None) -> None:
        super().__init__()
        self._levels = list(levels)
        self._current = current
        self._list = _ResultSelectList(
            self._levels,
            current=current,
            enable_search=False,
            max_height=12,
            on_selected=lambda item: self.dismiss(item.value if item is not None else None),
            on_cancelled=lambda: self.dismiss(None),
        )
        self._body = Vertical()
        self._body.mount(Label("Thinking level", height=1))
        self._body.mount(self._list)

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


class OAuthSelector(OverlayDialog):
    """OAuth provider 选择器（登录/登出）。"""

    def __init__(
        self,
        providers: list[tuple[str, str, bool]],
        *,
        mode: str = "login",
    ) -> None:
        super().__init__()
        self._providers = list(providers)
        self._mode = mode
        items = [
            SelectItem(
                value=provider_id,
                label=f"{name} ({provider_id}) [{'logged in' if logged_in else 'not logged in'}]",
            )
            for provider_id, name, logged_in in self._providers
        ]
        title = "Login provider" if mode == "login" else "Logout provider"
        self._list = _ResultSelectList(
            items,
            enable_search=False,
            max_height=12,
            on_selected=lambda item: self.dismiss(item.value if item is not None else None),
            on_cancelled=lambda: self.dismiss(None),
        )
        self._body = Vertical()
        self._body.mount(Label(title, height=1))
        self._body.mount(self._list)

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


class ScopedModelsSelector(OverlayDialog):
    """模型范围选择器：Enter 切换选中，ctrl+a 全选，ctrl+x 全清，
    ctrl+p 切换 provider，alt+up/down 调整顺序，ctrl+s 持久化，Esc 保存。

    选中顺序即模型循环顺序（对齐 TS scoped-models-selector）。
    """

    def __init__(
        self,
        models: list[Any],
        selected: set[tuple[str, str]] | None = None,
        current: Any | None = None,
        on_persist=None,
    ) -> None:
        super().__init__()
        self._models = list(models)
        self._selected = set(selected or {})
        self._order: list[tuple[str, str]] = [
            key for key in (selected or {}) if key in {self._key(m) for m in self._models}
        ]
        self._current = current
        self._selected_index = 0
        self._on_persist = on_persist

    def _key(self, model) -> tuple[str, str]:
        return (model.provider, model.id)

    def _provider_keys(self, provider: str) -> list[tuple[str, str]]:
        return [self._key(m) for m in self._models if m.provider == provider]

    def _toggle(self, model_key: tuple[str, str]) -> None:
        if model_key in self._selected:
            self._selected.discard(model_key)
            self._order.remove(model_key)
        else:
            self._selected.add(model_key)
            self._order.append(model_key)

    def _dismiss(self, persist: bool) -> None:
        if self._on_persist is not None and persist:
            self._on_persist(list(self._order))
        self.dismiss(list(self._order))

    def handle_key(self, key: Key) -> bool:
        name = key.name
        if name == "up":
            self._selected_index = (self._selected_index - 1) % len(self._models)
            self.refresh()
            return True
        if name == "down":
            self._selected_index = (self._selected_index + 1) % len(self._models)
            self.refresh()
            return True
        if name == "enter":
            if 0 <= self._selected_index < len(self._models):
                self._toggle(self._key(self._models[self._selected_index]))
                self.refresh()
            return True
        if name == "ctrl+a":
            # 全选（对齐 TS app.models.enableAll）。
            for model in self._models:
                model_key = self._key(model)
                if model_key not in self._selected:
                    self._selected.add(model_key)
                    self._order.append(model_key)
            self.refresh()
            return True
        if name == "ctrl+x":
            # 全清（对齐 TS app.models.clearAll）。
            self._selected.clear()
            self._order.clear()
            self.refresh()
            return True
        if name == "ctrl+p":
            # 切换当前 provider 的全部模型（对齐 TS app.models.toggleProvider）。
            if 0 <= self._selected_index < len(self._models):
                provider_keys = self._provider_keys(self._models[self._selected_index].provider)
                if all(k in self._selected for k in provider_keys):
                    for model_key in provider_keys:
                        if model_key in self._selected:
                            self._selected.discard(model_key)
                            self._order.remove(model_key)
                else:
                    for model_key in provider_keys:
                        if model_key not in self._selected:
                            self._selected.add(model_key)
                            self._order.append(model_key)
                self.refresh()
            return True
        if name in ("alt+up", "alt+down"):
            # 已启用顺序调整（对齐 TS app.models.reorderUp/reorderDown）。
            if 0 <= self._selected_index < len(self._models):
                model_key = self._key(self._models[self._selected_index])
                if model_key in self._order:
                    index = self._order.index(model_key)
                    delta = -1 if name == "alt+up" else 1
                    new_index = index + delta
                    if 0 <= new_index < len(self._order):
                        self._order[index], self._order[new_index] = (
                            self._order[new_index],
                            self._order[index],
                        )
                        self.refresh()
            return True
        if name == "ctrl+s":
            self._dismiss(persist=True)
            return True
        if name == "escape":
            self._dismiss(persist=False)
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        title = (
            "Scoped models (Enter: toggle, ctrl+a: all, ctrl+x: clear, "
            "ctrl+p: provider, alt+up/down: reorder, ctrl+s: save, Esc: close)"
        )
        lines: list[Line] = [line_from_text(title, width)]
        visible = min(height - 1, len(self._models))
        for index in range(visible):
            model = self._models[index]
            model_key = self._key(model)
            position = ""
            if model_key in self._order:
                position = f" #{self._order.index(model_key) + 1}"
            check = " ✓" if model_key in self._selected else ""
            marker = ">" if index == self._selected_index else " "
            style = None
            if index == self._selected_index:
                from rich.style import Style

                style = Style(reverse=True)
            lines.append(
                line_from_text(
                    f"{marker} {model.provider}/{model.id}{position}{check}", width, style
                )
            )
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines

    def content_size(self) -> tuple[int, int]:
        return (1000, min(1 + len(self._models), 16))


class ExtensionSelector(OverlayDialog):
    """扩展列表选择器。"""

    def __init__(self, extensions: list[dict[str, Any]]) -> None:
        super().__init__()
        self._extensions = list(extensions)
        items = [
            SelectItem(
                value=extension.get("path", str(index)),
                label=extension.get("label", extension.get("path", "?")),
            )
            for index, extension in enumerate(self._extensions)
        ]
        self._list = _ResultSelectList(
            items,
            enable_search=True,
            max_height=12,
            on_selected=self._on_selected,
            on_cancelled=lambda: self.dismiss(None),
        )
        self._body = Vertical()
        self._body.mount(Label("Extensions", height=1))
        self._body.mount(self._list)

    def _on_selected(self, item: SelectItem | None) -> None:
        if item is None:
            self.dismiss(None)
            return
        for extension in self._extensions:
            if extension.get("path", "") == item.value:
                self.dismiss(extension)
                return
        self.dismiss(None)

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


class TrustSelector(OverlayDialog):
    """项目信任选择器。

    `options` 由调用方（pi_coding_agent）通过 get_project_trust_options 注入，
    保持 pi_tui 不反向依赖 pi_coding_agent。
    """

    def __init__(
        self,
        cwd: str,
        options: list[dict],
        saved_decision: dict | None = None,
        project_trusted: bool = False,
    ) -> None:
        super().__init__()
        self._cwd = cwd
        self._saved_decision = saved_decision
        self._project_trusted = project_trusted
        self._options = options
        self._selected = 0
        if saved_decision is not None:
            for index, option in enumerate(self._options):
                if option.get("savedPath") == saved_decision.get("path") and option.get(
                    "trusted"
                ) == saved_decision.get("decision"):
                    self._selected = index
                    break
        items = [
            SelectItem(
                value=option.get("label", ""),
                label=option.get("label", ""),
            )
            for option in self._options
        ]
        self._list = _ResultSelectList(
            items,
            enable_search=False,
            max_height=12,
            on_selected=self._on_selected,
            on_cancelled=lambda: self.dismiss(None),
        )
        self._list._selected_index = self._selected
        self._body = Vertical()
        self._body.mount(Label("Project trust", height=1))
        self._body.mount(Label(self._cwd, height=1))
        status = self._format_decision()
        self._body.mount(
            Label(
                f"Saved decision: {status}  |  Current session: "
                f"{'trusted' if self._project_trusted else 'untrusted'}",
                height=1,
            )
        )
        self._body.mount(self._list)

    def _format_decision(self) -> str:
        entry = self._saved_decision
        if entry is None:
            return "none"
        label = "trusted" if entry.get("decision") else "untrusted"
        if entry.get("path") != self._cwd:
            return f"{label} (inherited from {entry.get('path')})"
        return f"{label} ({entry.get('path')})"

    def _on_selected(self, item: SelectItem | None) -> None:
        if item is None:
            self.dismiss(None)
            return
        for option in self._options:
            if option.get("label", "") == item.value:
                self.dismiss(option)
                return
        self.dismiss(None)

    def handle_key(self, key: Key) -> bool:
        return self._list.handle_key(key)

    def render(self, width: int, height: int) -> list[Line]:
        return self._body.render(width, height)

    def content_size(self) -> tuple[int, int]:
        return self._body.content_size()


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
