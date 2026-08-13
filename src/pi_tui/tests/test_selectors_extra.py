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
    assert "faux-1 [faux]" in "\n".join(line.text() for line in lines)
    assert selector.handle_key(_key("enter")) is True


def test_model_selector_scroll_window_and_current_marker() -> None:
    """滚动窗口居中、滚动指示器、当前模型 ✓ 标记（对齐 TS model-selector）。"""
    models = [SimpleNamespace(provider="faux", id=f"m-{index}", name="M") for index in range(30)]
    # 当前模型指向列表尾部：打开时窗口应包含它（居中滚动）。
    selector = ModelSelector(models, current=models[25])
    lines = selector.render(80, 12)
    text = "\n".join(line.text() for line in lines)
    assert "m-25 [faux]" in text  # 当前模型在窗口内
    assert "✓" in text  # 当前标记
    assert "(26/30)" in text  # 滚动指示器

    # 导航到末尾：窗口跟随且指示器更新
    for _ in range(4):
        selector.handle_key(_key("down"))
    text = "\n".join(line.text() for line in selector.render(80, 12))
    assert "m-29 [faux]" in text
    assert "(30/30)" in text


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


def test_choice_and_input_dialog_update_title() -> None:
    """ChoiceSelector / TextInputDialog 的 update_title（倒计时文案）。"""
    choice = ChoiceSelector("Pick", ["a", "b"])
    choice.update_title("Pick (auto-cancel in 3s)")
    text = "\n".join(line.text() for line in choice.render(60, 4))
    assert "auto-cancel in 3s" in text

    dialog = TextInputDialog("Name?", placeholder="")
    dialog.update_title("Name? (auto-cancel in 2s)")
    text = "\n".join(line.text() for line in dialog.render(60, 4))
    assert "auto-cancel in 2s" in text


def test_scoped_models_selector_toggle_and_save() -> None:
    models = [SimpleNamespace(provider="faux", id="faux-1")]
    selector = ScopedModelsSelector(models, selected=set())
    assert selector.handle_key(_key("enter")) is True
    assert ("faux", "faux-1") in selector._selected
    assert selector.handle_key(_key("escape")) is True
    lines = selector.render(80, 20)
    assert "✓" in "\n".join(line.text() for line in lines)


def test_scoped_models_selector_bulk_ops_and_reorder() -> None:
    """ctrl+a 全选、ctrl+x 全清、ctrl+p 切换 provider、alt+up/down 重排、ctrl+s 持久化。"""
    models = [
        SimpleNamespace(provider="faux", id="faux-1"),
        SimpleNamespace(provider="faux", id="faux-2"),
        SimpleNamespace(provider="openai", id="gpt-1"),
    ]
    selector = ScopedModelsSelector(models, selected=set())
    assert selector.handle_key(_key("ctrl+a")) is True
    assert len(selector._selected) == 3
    assert len(selector._order) == 3

    # ctrl+p：当前行（index 0，faux）已全选 → 清空 faux 全部
    assert selector.handle_key(_key("ctrl+p")) is True
    assert selector._selected == {("openai", "gpt-1")}

    # 全清
    assert selector.handle_key(_key("ctrl+x")) is True
    assert selector._selected == set()

    # 重排：选中 faux-1、faux-2、gpt-1 后 alt+up 调整 gpt-1（index 2）到前
    selector.handle_key(_key("ctrl+a"))
    selector._selected_index = 2
    assert selector.handle_key(_key("alt+up")) is True
    assert selector._order == [("faux", "faux-1"), ("openai", "gpt-1"), ("faux", "faux-2")]

    # ctrl+s 持久化回调
    persisted: list = []
    selector._on_persist = persisted.append
    assert selector.handle_key(_key("ctrl+s")) is True
    assert persisted == [selector._order]


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
