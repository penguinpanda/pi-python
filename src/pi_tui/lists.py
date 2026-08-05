"""通用列表组件：SelectList / SettingsList（对齐 TS 的 Python 子集）。

- SelectList：可选模糊搜索，上下键导航，Enter 选择 / Escape 取消。
- SettingsList：label + 当前值两列，Enter 循环取值（values 提供时）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView


@dataclass(frozen=True)
class SelectItem:
    """可选项。"""

    value: str
    label: str | None = None
    description: str | None = None

    @property
    def display_label(self) -> str:
        return self.label or self.value


@dataclass(frozen=True)
class SettingItem:
    """设置项：label + 当前值；values 提供时 Enter 循环取值。"""

    id: str
    label: str
    current_value: str = ""
    values: list[str] | None = None
    description: str | None = None


def _normalize_select_items(items: Sequence[SelectItem | str]) -> list[SelectItem]:
    return [item if isinstance(item, SelectItem) else SelectItem(value=str(item)) for item in items]


def _filter_score(text: str, query: str) -> int:
    """模糊匹配打分：2=前缀，1=子串，0=子序列，-1=不匹配。"""
    text = text.lower()
    query = query.strip().lower()
    if not query:
        return 0
    if text.startswith(query):
        return 2
    if query in text:
        return 1
    iterator = iter(text)
    if all(any(ch == wanted for ch in iterator) for wanted in query):
        return 0
    return -1


def _filter_items(items: list[SelectItem], query: str) -> list[SelectItem]:
    scored: list[tuple[int, SelectItem]] = []
    for item in items:
        score = max(
            _filter_score(item.value, query),
            _filter_score(item.display_label, query),
        )
        if score >= 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].display_label.lower()))
    return [item for _, item in scored]


class SelectList(Widget):
    """可筛选选项列表。"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    SelectList {
        height: auto;
        max-height: 14;
    }
    SelectList ListView {
        height: auto;
        max-height: 10;
    }
    .select-list-search {
        margin: 0 0 1 0;
    }
    """

    class Selected(Message):
        """Enter 选择了一项。"""

        def __init__(self, item: SelectItem) -> None:
            super().__init__()
            self.item = item

    class Cancelled(Message):
        """Escape 取消。"""

        pass

    def __init__(
        self,
        items: Sequence[SelectItem | str],
        *,
        current: str | None = None,
        enable_search: bool = True,
        search_placeholder: str = "Filter...",
        list_id: str | None = None,
    ) -> None:
        super().__init__()
        self._items = _normalize_select_items(items)
        self._filtered = list(self._items)
        self._current = current
        self._enable_search = enable_search
        self._search_placeholder = search_placeholder
        self._list_id = list_id or "select-list-view"
        self._selected_index = 0
        if current is not None:
            for index, item in enumerate(self._items):
                if item.value == current:
                    self._selected_index = index
                    break

    @property
    def filtered_items(self) -> list[SelectItem]:
        return list(self._filtered)

    @property
    def selected_item(self) -> SelectItem | None:
        if self.is_mounted:
            view = self.query_one(f"#{self._list_id}", ListView)
            index = view.index
            if index is not None and 0 <= index < len(self._filtered):
                return self._filtered[index]
            return None
        if 0 <= self._selected_index < len(self._filtered):
            return self._filtered[self._selected_index]
        return None

    def compose(self) -> ComposeResult:
        with Vertical():
            if self._enable_search:
                yield Input(
                    placeholder=self._search_placeholder,
                    classes="select-list-search",
                )
            yield ListView(id=self._list_id)

    async def on_mount(self) -> None:
        await self._rebuild_list()
        self._focus_list()

    async def on_input_changed(self, event: Input.Changed) -> None:
        await self._apply_filter(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # 搜索框 Enter：把焦点交回列表，不触发选择。
        self._focus_list()

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        view = self.query_one(f"#{self._list_id}", ListView)
        index = view.index
        if index is not None and 0 <= index < len(self._filtered):
            self._selected_index = index
            self.post_message(self.Selected(self._filtered[index]))
        else:
            self.post_message(self.Cancelled())

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())

    async def _apply_filter(self, query: str) -> None:
        self._filtered = _filter_items(self._items, query)
        self._selected_index = 0
        await self._rebuild_list()

    async def _rebuild_list(self) -> None:
        view = self.query_one(f"#{self._list_id}", ListView)
        await view.clear()
        for item in self._filtered:
            marker = ">" if item.value == self._current else " "
            description = f"  [dim]{item.description}[/dim]" if item.description else ""
            await view.append(ListItem(Label(f"{marker} {item.display_label}{description}")))
        if self._filtered:
            view.index = self._selected_index

    def _focus_list(self) -> None:
        if self._filtered:
            self.query_one(f"#{self._list_id}", ListView).focus()


class SettingsList(SelectList):
    """设置项列表：Enter 循环取值（values 提供时），否则仅激活。"""

    class Changed(Message):
        """设置项值发生变化。"""

        def __init__(self, item: SettingItem, value: str) -> None:
            super().__init__()
            self.item = item
            self.value = value

    class Activated(Message):
        """无 values 的项被激活（预留子菜单）。"""

        def __init__(self, item: SettingItem) -> None:
            super().__init__()
            self.item = item

    def __init__(
        self,
        items: Sequence[SettingItem],
        *,
        enable_search: bool = False,
        list_id: str | None = None,
    ) -> None:
        self._setting_items = list(items)
        self._values = {item.id: item.current_value for item in self._setting_items}
        super().__init__(
            [item.label for item in self._setting_items],
            enable_search=enable_search,
            list_id=list_id,
        )

    def values(self) -> dict[str, str]:
        return dict(self._values)

    def _item_at(self, index: int) -> SettingItem | None:
        if 0 <= index < len(self._filtered):
            label = self._filtered[index].value
            for item in self._setting_items:
                if item.label == label:
                    return item
        return None

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        view = self.query_one(f"#{self._list_id}", ListView)
        index = view.index
        if index is None or not (0 <= index < len(self._filtered)):
            self.post_message(self.Cancelled())
            return
        item = self._item_at(index)
        if item is None:
            return
        if item.values:
            values = item.values
            current = self._values.get(item.id, item.current_value)
            next_index = values.index(current) + 1 if current in values else 0
            next_value = values[next_index % len(values)]
            self._values[item.id] = next_value
            self.post_message(self.Changed(item, next_value))
        else:
            self.post_message(self.Activated(item))
        await self._rebuild_list()

    async def _rebuild_list(self) -> None:
        view = self.query_one(f"#{self._list_id}", ListView)
        await view.clear()
        for item in self._filtered:
            setting = self._item_at(self._filtered.index(item))
            value = self._values.get(setting.id, setting.current_value) if setting else ""
            label = f"{item.display_label}  [dim]{value}[/dim]"
            await view.append(ListItem(Label(label)))
        if self._filtered:
            view.index = self._selected_index
