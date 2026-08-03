"""TUI 快捷键系统测试。"""

from __future__ import annotations

from pi_tui.keybindings import Keybinding, KeybindingsManager


class TestKeybindingsManager:
    def test_defaults(self):
        manager = KeybindingsManager()
        assert manager.resolve("ctrl+p") == "app.model.cycleForward"
        assert manager.resolve("escape") == "app.interrupt"
        assert manager.resolve("shift+tab") == "app.thinking.cycle"

    def test_register_override(self):
        manager = KeybindingsManager()
        manager.register(
            Keybinding("ctrl+k", "app.model.cycleForward", "cycle_model_forward", "Custom")
        )
        assert manager.resolve("ctrl+k") == "app.model.cycleForward"
        assert manager.resolve("ctrl+p") is None

    def test_user_bindings_string(self):
        manager = KeybindingsManager(user_bindings={"app.model.cycleForward": "ctrl+k"})
        assert manager.resolve("ctrl+k") == "app.model.cycleForward"
        assert manager.resolve("ctrl+p") is None
        assert manager.get_action_key("app.model.cycleForward") == "ctrl+k"

    def test_user_bindings_list(self):
        manager = KeybindingsManager(
            user_bindings={"app.model.cycleForward": ["ctrl+k", "ctrl+j"]}
        )
        assert manager.get_action_key("app.model.cycleForward") == "ctrl+k"
        assert manager.resolve("ctrl+j") == "app.model.cycleForward"

    def test_user_bindings_disable(self):
        manager = KeybindingsManager(
            user_bindings={"app.model.cycleForward": []}
        )
        assert manager.is_enabled("app.model.cycleForward") is False
        assert manager.resolve("ctrl+p") is None
        assert all(
            binding.action_id != "app.model.cycleForward"
            for binding in manager.all_bindings()
        )

    def test_load_from_settings(self):
        manager = KeybindingsManager()
        manager.load_from_settings(
            {"keybindings": {"app.model.select": "ctrl+m"}}
        )
        assert manager.resolve("ctrl+m") == "app.model.select"
        assert manager.resolve("ctrl+l") is None

    def test_load_from_settings_ignores_unknown(self):
        manager = KeybindingsManager()
        manager.load_from_settings({"keybindings": {"app.nope": "ctrl+z"}})
        assert manager.resolve("ctrl+z") is None

    def test_all_bindings_actions(self):
        manager = KeybindingsManager()
        actions = {binding.action for binding in manager.all_bindings()}
        assert "cycle_model_forward" in actions
        assert "exit" in actions

    def test_default_keys_present(self):
        manager = KeybindingsManager()
        assert manager.get_action_key("app.exit") == "ctrl+d"
        assert manager.get_action_key("app.message.followUp") == "alt+enter"
        assert manager.get_action_key("app.session.new") == "ctrl+n"
