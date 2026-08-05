"""SelectList / SettingsList 组件测试。"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, Label, ListView

from pi_tui.lists import SelectItem, SelectList, SettingItem, SettingsList


class _ListHarness(App):
    def __init__(self, widget) -> None:
        super().__init__()
        self._widget = widget
        self.selected: str | None = None
        self.cancelled = False
        self.changed: list[tuple[str, str]] = []

    def compose(self):
        yield self._widget

    def on_select_list_selected(self, event) -> None:
        self.selected = event.item.value

    def on_select_list_cancelled(self, event) -> None:
        self.cancelled = True

    def on_settings_list_changed(self, event) -> None:
        self.changed.append((event.item.id, event.value))


@pytest.mark.asyncio
async def test_select_list_navigation_and_select() -> None:
    app = _ListHarness(SelectList(["a", "b", "c"], current="b"))
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one("#select-list-view", ListView)
        assert view.index == 1
        await pilot.press("down")
        assert view.index == 2
        await pilot.press("enter")
        await pilot.pause()
        assert app.selected == "c"


@pytest.mark.asyncio
async def test_select_list_filter_and_select_first_match() -> None:
    app = _ListHarness(SelectList(["alpha", "beta", "alpine"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(Input).focus()
        await pilot.pause()
        await pilot.press("a", "l", "p")
        await pilot.pause()
        select = app.query_one(SelectList)
        assert [item.value for item in select.filtered_items] == ["alpha", "alpine"]
        view = app.query_one("#select-list-view", ListView)
        assert len(view.children) == 2
        await pilot.press("enter")
        await pilot.pause()
        # 搜索框 Enter 只把焦点交回列表；再 Enter 才选择。
        assert app.focused is view
        await pilot.press("enter")
        await pilot.pause()
        assert app.selected == "alpha"


@pytest.mark.asyncio
async def test_select_list_cancel() -> None:
    app = _ListHarness(SelectList(["a", "b"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.cancelled is True


@pytest.mark.asyncio
async def test_select_item_description_and_current_marker() -> None:
    items = [
        SelectItem(value="one", description="first"),
        SelectItem(value="two"),
    ]
    app = _ListHarness(SelectList(items, current="two"))
    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one("#select-list-view", ListView)
        assert view.index == 1
        labels = [child.query_one(Label).render().plain for child in view.children]
        assert "> two" in labels[1]


@pytest.mark.asyncio
async def test_settings_list_cycles_values() -> None:
    items = [
        SettingItem(
            id="level",
            label="Level",
            current_value="low",
            values=["low", "medium", "high"],
        )
    ]
    app = _ListHarness(SettingsList(items, list_id="settings-list"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.changed == [("level", "medium")]
        await pilot.press("enter")
        await pilot.pause()
        assert app.changed[-1] == ("level", "high")
        settings = app.query_one(SettingsList)
        assert settings.values() == {"level": "high"}


@pytest.mark.asyncio
async def test_settings_list_activated_without_values() -> None:
    items = [SettingItem(id="open", label="Open", current_value="")]
    app = _ListHarness(SettingsList(items, list_id="settings-list"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.changed == []
