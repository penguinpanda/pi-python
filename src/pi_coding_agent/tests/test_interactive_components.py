"""coding-agent 交互组件测试。"""

from __future__ import annotations

from pi_coding_agent.modes.interactive.components import (
    ArminComponent,
    ConfigSelectorComponent,
    DaxnutsComponent,
    EarendilAnnouncementComponent,
    FirstTimeSetupComponent,
    LoginDialogComponent,
    ShowImagesSelectorComponent,
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
    toggled: list[tuple[dict, bool]] = []
    entry = {"resource_type": "skills", "path": "a", "enabled": True}
    component = ConfigSelectorComponent(
        [entry], lambda item, enabled: toggled.append((item, enabled)), lambda: None
    )
    assert component.handle_key(_key("enter")) is True
    assert entry["enabled"] is False
    assert toggled == [(entry, False)]


def test_art_components_render() -> None:
    for component in (ArminComponent(), DaxnutsComponent(), EarendilAnnouncementComponent()):
        lines = component.render(60, 8)
        assert lines


def test_diff_lines_color_removed_and_added() -> None:
    lines = render_diff_lines("-1 old\n+1 new", 40)
    text = "\n".join(line.text() for line in lines)
    assert "-1 old" in text
    assert "+1 new" in text
