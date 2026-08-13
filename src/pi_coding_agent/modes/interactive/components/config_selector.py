"""交互式资源配置选择器。

组件负责展示资源树、搜索、键盘导航和切换动作；实际路径发现和设置持久化由
`pi_coding_agent.config_selector` 提供。这样组件保持纯 UI，应用层决定资源语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.keys import Key
from pi_tui.engine.widgets import Input, Widget


ResourceType = Literal["extensions", "skills", "prompts", "themes"]
ResourceScope = Literal["user", "project"]
ResourceOrigin = Literal["top-level", "package"]
ConfigScope = Literal["global", "project"]


@dataclass(slots=True)
class ResourceItem:
    key: str
    resource_type: ResourceType
    path: str
    enabled: bool
    scope: ResourceScope
    origin: ResourceOrigin
    source: str
    display_name: str
    base_dir: str


@dataclass(slots=True)
class ResourceGroup:
    key: str
    label: str
    scope: ResourceScope
    origin: ResourceOrigin
    source: str
    items: list[ResourceItem] = field(default_factory=list)


@dataclass(slots=True)
class ConfigSelectorModel:
    """配置选择器数据模型。"""

    groups: list[ResourceGroup]
    cwd: str
    agent_dir: str
    write_scope: ConfigScope
    project_mode_available: bool

    def all_items(self) -> list[ResourceItem]:
        return [item for group in self.groups for item in group.items]


@dataclass(slots=True)
class _FlatEntry:
    kind: Literal["group", "item"]
    group: ResourceGroup | None = None
    item: ResourceItem | None = None


class ConfigSelectorComponent(Widget):
    """搜索、分组展示和切换本地资源。"""

    def __init__(
        self,
        model: ConfigSelectorModel,
        *,
        on_toggle: Callable[[ResourceItem, bool], None],
        on_close: Callable[[], None],
        on_exit: Callable[[], None],
        on_switch_scope: Callable[[], None],
    ) -> None:
        super().__init__(focusable=True)
        self.model = model
        self._on_toggle = on_toggle
        self._on_close = on_close
        self._on_exit = on_exit
        self._on_switch_scope = on_switch_scope
        self._search = Input(placeholder="Search resources...")
        self._selected = 0
        self._visible: list[_FlatEntry] = []
        self._rebuild()

    def _rebuild(self) -> None:
        query = self._search.value.strip().lower()
        entries: list[_FlatEntry] = []
        for group in self.model.groups:
            visible_items = [
                item
                for item in group.items
                if not query
                or query in item.display_name.lower()
                or query in item.resource_type.lower()
                or query in item.path.lower()
            ]
            if visible_items:
                entries.append(_FlatEntry("group", group=group))
                entries.extend(_FlatEntry("item", item=item) for item in visible_items)
        self._visible = entries
        if not self._visible:
            self._selected = 0
            return
        item_indexes = [index for index, entry in enumerate(entries) if entry.kind == "item"]
        self._selected = next(
            (index for index in item_indexes if index >= self._selected),
            item_indexes[0],
        )

    def _next_item(self, delta: int) -> None:
        if not self._visible:
            return
        index = self._selected + delta
        while 0 <= index < len(self._visible):
            if self._visible[index].kind == "item":
                self._selected = index
                self.refresh()
                return
            index += delta

    def _selected_item(self) -> ResourceItem | None:
        entry = self._visible[self._selected] if self._visible else None
        return entry.item if entry is not None and entry.kind == "item" else None

    def _toggle_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        enabled = not item.enabled
        self._on_toggle(item, enabled)
        item.enabled = enabled
        self.refresh()

    def handle_key(self, key: Key) -> bool:
        if key.name == "escape":
            self._on_close()
            return True
        if key.name == "ctrl+c":
            self._on_exit()
            return True
        if key.name == "up" or key.name == "k":
            self._next_item(-1)
            return True
        if key.name == "down" or key.name == "j":
            self._next_item(1)
            return True
        if key.name in ("enter", " "):
            self._toggle_selected()
            return True
        if key.name == "tab":
            self._on_switch_scope()
            return True
        if self._search.handle_key(key):
            self._rebuild()
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = [line_from_text("Local resources", width)]
        lines.extend(self._search.render(width, 1))
        lines.append(blank_line(width))
        if not self._visible:
            lines.append(line_from_text("  No resources found", width))
        else:
            start = max(0, self._selected - max(0, height - 8))
            end = min(len(self._visible), start + max(1, height - 4))
            for index in range(start, end):
                entry = self._visible[index]
                marker = ">" if index == self._selected else " "
                if entry.kind == "group" and entry.group is not None:
                    lines.append(line_from_text(f"  {entry.group.label}", width))
                elif entry.item is not None:
                    item = entry.item
                    checkbox = "[x]" if item.enabled else "[ ]"
                    suffix = f"  {item.source}" if item.origin == "package" else ""
                    lines.append(
                        line_from_text(
                            f"{marker} {checkbox} {item.display_name}{suffix}",
                            width,
                        )
                    )
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (80, min(4 + len(self._visible), 24))


__all__ = [
    "ConfigScope",
    "ConfigSelectorComponent",
    "ConfigSelectorModel",
    "ResourceGroup",
    "ResourceItem",
    "ResourceOrigin",
    "ResourceScope",
    "ResourceType",
]
