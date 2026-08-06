"""引擎版 SelectList / SettingsList 测试。"""

from __future__ import annotations

from pi_tui.engine import App, FakeTerminal, SelectItem, SelectList, SettingItem, SettingsList
from pi_tui.engine.keys import Key, parse_input


def _key(name: str) -> Key:
    return Key(name)


def _char(char: str) -> Key:
    return parse_input(char.encode("utf-8"))[0].key


def test_select_list_navigation_and_select() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    selected: list[SelectItem] = []
    app.on_select_list_selected = lambda message: selected.append(message.item)  # type: ignore[method-assign]
    listing = SelectList(["a", "b", "c"])
    listing.app = app
    listing.handle_key(_key("down"))
    listing.handle_key(_key("down"))
    listing.handle_key(_key("enter"))
    assert selected and selected[0].value == "c"


def test_select_list_filter_and_select_first_match() -> None:
    listing = SelectList(["alpha", "beta", "gamma"])
    listing.handle_key(_char("b"))
    assert [item.value for item in listing.filtered_items] == ["beta"]
    listing.handle_key(_key("enter"))
    assert listing.selected_item is not None and listing.selected_item.value == "beta"


def test_select_list_cancel() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    cancelled: list[bool] = []
    app.on_select_list_cancelled = lambda _message: cancelled.append(True)  # type: ignore[method-assign]
    listing = SelectList(["a"])
    listing.app = app
    listing.handle_key(_key("escape"))
    assert cancelled == [True]


def test_select_item_description_and_current_marker() -> None:
    item = SelectItem("v", label="Label", description="desc")
    assert item.display_label == "Label"
    listing = SelectList([item], current="v")
    lines = listing.render(30, 5)
    assert "desc" in lines[1].text()
    assert "Label" in lines[1].text()


def test_settings_list_cycles_values() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    changes: list[tuple[str, str]] = []
    app.on_settings_list_changed = (  # type: ignore[method-assign]
        lambda message: changes.append((message.item.id, message.value))
    )
    settings = SettingsList(
        [
            SettingItem(
                id="mode", label="Mode", current_value="ask", values=["ask", "trust", "block"]
            )
        ]
    )
    settings.app = app
    settings.handle_key(_key("enter"))
    settings.handle_key(_key("enter"))
    assert changes == [("mode", "trust"), ("mode", "block")]
    assert settings.values() == {"mode": "block"}


def test_settings_list_activated_without_values() -> None:
    app = App(terminal=FakeTerminal(size=(40, 12)))
    activated: list[str] = []
    app.on_settings_list_activated = (  # type: ignore[method-assign]
        lambda message: activated.append(message.item.id)
    )
    settings = SettingsList([SettingItem(id="sub", label="Submenu")])
    settings.app = app
    settings.handle_key(_key("enter"))
    assert activated == ["sub"]
