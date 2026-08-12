"""pi_tui.selectors 补充测试。"""

from __future__ import annotations

from types import SimpleNamespace

from pi_tui.engine.keys import Key
from pi_tui.selectors import (
    ChoiceSelector,
    ExtensionSelector,
    ModelSelector,
    OAuthSelector,
    ScopedModelsSelector,
    SettingsSelector,
    TextInputDialog,
    ThinkingSelector,
    TrustSelector,
)


def _key(name: str, char: str | None = None) -> Key:
    return Key(name=name, char=char)


def test_model_selector_navigation_and_render() -> None:
    models = [
        SimpleNamespace(provider="faux", id="faux-1", name="Faux"),
        SimpleNamespace(provider="openai", id="gpt-5", name="GPT"),
    ]
    selector = ModelSelector(models, current=models[0])
    assert selector.handle_key(_key("down")) is True
    lines = selector.render(80, 20)
    assert "faux/faux-1" in "\n".join(line.text() for line in lines)
    assert selector.handle_key(_key("enter")) is True


def test_choice_selector_search() -> None:
    selector = ChoiceSelector("Pick", ["alpha", "beta"])
    assert selector.handle_key(_key("b", "b")) is True
    labels = [item.display_label for item in selector._list.filtered_items]
    assert labels == ["beta"]
    lines = selector.render(80, 20)
    assert "beta" in "\n".join(line.text() for line in lines)


def test_settings_selector_bool_and_navigation() -> None:
    changes: list[tuple[str, object]] = []
    selector = SettingsSelector(
        [
            {"key": "enabled", "label": "Enabled", "type": "bool"},
            {"key": "name", "label": "Name", "type": "string"},
        ],
        {"enabled": False, "name": "x"},
        lambda key, value: changes.append((key, value)),
    )
    assert selector.handle_key(_key("down")) is True
    assert selector.handle_key(_key("up")) is True
    assert selector.handle_key(_key("enter")) is True
    assert changes == [("enabled", True)]
    lines = selector.render(80, 20)
    assert "Enabled: true" in "\n".join(line.text() for line in lines)


def test_thinking_and_oauth_selectors() -> None:
    thinking = ThinkingSelector(["off", "high"], current="high")
    assert thinking.handle_key(_key("down")) is True
    assert "Thinking level" in "\n".join(line.text() for line in thinking.render(80, 20))

    oauth = OAuthSelector([("openai", "OpenAI", True)])
    assert oauth.handle_key(_key("enter")) is True
    text = "\n".join(line.text() for line in oauth.render(80, 20))
    assert "OpenAI" in text


def test_scoped_models_selector_toggle_and_save() -> None:
    models = [SimpleNamespace(provider="faux", id="faux-1")]
    selector = ScopedModelsSelector(models, selected=set())
    assert selector.handle_key(_key("enter")) is True
    assert ("faux", "faux-1") in selector._selected
    assert selector.handle_key(_key("escape")) is True
    lines = selector.render(80, 20)
    assert "✓" in "\n".join(line.text() for line in lines)


def test_extension_selector_search() -> None:
    selector = ExtensionSelector(
        [
            {"path": "/tmp/a.py", "label": "Alpha"},
            {"path": "/tmp/b.py", "label": "Beta"},
        ]
    )
    assert selector.handle_key(_key("b", "b")) is True
    assert [item.value for item in selector._list.filtered_items] == ["/tmp/b.py"]


def test_trust_selector_render() -> None:
    selector = TrustSelector(
        "/tmp/project",
        options=[
            {"label": "Ask every time", "trusted": False},
            {"label": "Trust", "trusted": True},
        ],
    )
    text = "\n".join(line.text() for line in selector.render(80, 20))
    assert "Project trust" in text


def test_text_input_dialog() -> None:
    dialog = TextInputDialog("Enter name", value="abc")
    assert dialog.handle_key(_key("d", "d")) is True
    assert dialog._input.value == "abcd"
    assert dialog.handle_key(_key("escape")) is True
