"""coding-agent 交互组件测试。"""

from __future__ import annotations

from pi_coding_agent.modes.interactive.components import (
    ArminComponent,
    ConfigSelectorComponent,
    ConfigSelectorModel,
    DaxnutsComponent,
    EarendilAnnouncementComponent,
    FirstTimeSetupComponent,
    LoginDialogComponent,
    ShowImagesSelectorComponent,
    ResourceGroup,
    ResourceItem,
    render_diff_lines,
)
from pi_tui.engine.keys import Key


def _key(name: str) -> Key:
    return Key(name=name)


def test_first_time_setup_component_steps() -> None:
    previewed: list[str] = []
    submitted: list[tuple[str, bool]] = []
    component = FirstTimeSetupComponent(
        "dark",
        previewed.append,
        lambda theme, analytics: submitted.append((theme, analytics)),
        lambda: None,
    )
    assert component.handle_key(_key("down")) is True
    assert previewed == ["light"]
    assert component.handle_key(_key("enter")) is True
    assert component.handle_key(_key("down")) is True
    assert component.handle_key(_key("enter")) is True
    assert submitted == [("light", False)]


def test_login_dialog_shows_auth_and_device_code() -> None:
    completed: list[tuple[bool, str | None]] = []
    component = LoginDialogComponent("openai", lambda ok, msg: completed.append((ok, msg)))
    component.show_auth("https://auth.example", "Open in browser")
    text = "\n".join(line.text() for line in component.render(80, 8))
    assert "auth.example" in text
    component.show_device_code({"verificationUri": "https://device", "userCode": "1234"})
    text = "\n".join(line.text() for line in component.render(80, 8))
    assert "1234" in text


def test_show_images_selector_saves_choice() -> None:
    selected: list[bool] = []
    component = ShowImagesSelectorComponent(True, selected.append, lambda: None)
    assert component.handle_key(_key("down")) is True
    assert component.handle_key(_key("enter")) is True
    assert selected == [False]


def test_config_selector_toggles_entries() -> None:
    toggled: list[tuple[ResourceItem, bool]] = []
    item = ResourceItem(
        key="skills:a",
        resource_type="skills",
        path="a",
        enabled=True,
        scope="user",
        origin="top-level",
        source="user",
        display_name="a",
        base_dir="",
    )
    model = ConfigSelectorModel(
        groups=[ResourceGroup("skills", "user skills", "user", "top-level", "user", [item])],
        cwd="/tmp",
        agent_dir="/tmp/agent",
        write_scope="global",
        project_mode_available=False,
    )
    component = ConfigSelectorComponent(
        model,
        on_toggle=lambda item, enabled: toggled.append((item, enabled)),
        on_close=lambda: None,
        on_exit=lambda: None,
        on_switch_scope=lambda: None,
    )
    assert component.handle_key(_key("enter")) is True
    assert item.enabled is False
    assert toggled == [(item, False)]


def test_art_components_render() -> None:
    for component in (ArminComponent(), DaxnutsComponent(), EarendilAnnouncementComponent()):
        lines = component.render(60, 8)
        assert lines


def test_diff_lines_color_removed_and_added() -> None:
    lines = render_diff_lines("-1 old\n+1 new", 40)
    text = "\n".join(line.text() for line in lines)
    assert "-1 old" in text
    assert "+1 new" in text


def test_diff_lines_highlight_changed_word() -> None:
    lines = render_diff_lines("-1 keep old value\n+1 keep new value", 50)
    removed = lines[0]
    added = lines[1]
    removed_text = removed.text()
    added_text = added.text()
    assert "old" in removed_text
    assert "new" in added_text
    assert any(cell.style is not None and cell.style.reverse for cell in removed.cells)
    assert any(cell.style is not None and cell.style.reverse for cell in added.cells)


def test_diff_lines_falls_back_to_character_changes() -> None:
    lines = render_diff_lines("-1 abcdef\n+1 abXdef", 40)
    removed = lines[0].text()
    added = lines[1].text()
    assert "abcdef" in removed
    assert "abXdef" in added
    assert any(cell.style is not None and cell.style.reverse for cell in lines[1].cells)
