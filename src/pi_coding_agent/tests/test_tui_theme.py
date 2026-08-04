"""TUI 主题系统测试。"""

from __future__ import annotations

import json

import pytest

from pi_tui.theme import (
    COLOR_KEYS,
    BUILTIN_THEMES,
    Theme,
    ThemeError,
    ThemeLoader,
    validate_theme_colors,
)


class TestBuiltinThemes:
    def test_color_keys_present_in_builtins(self):
        for name, colors in BUILTIN_THEMES.items():
            validate_theme_colors(colors, name)

    def test_color_keys_count(self):
        assert len(COLOR_KEYS) >= 40

    def test_load_dark_and_light(self):
        loader = ThemeLoader()
        dark = loader.load("dark")
        light = loader.load("light")
        assert isinstance(dark, Theme)
        assert dark.colors["bg"] != light.colors["bg"]
        assert dark.colors["text"] != light.colors["text"]

    def test_css_variables(self):
        loader = ThemeLoader()
        theme = loader.load("dark")
        variables = theme.css_variables()
        assert "pi-bg" in variables
        assert variables["pi-bg"] == theme.colors["bg"]


class TestCustomTheme:
    def test_load_custom_json(self, tmp_path):
        colors = dict(BUILTIN_THEMES["dark"])
        colors["bg"] = "#000000"
        (tmp_path / "custom.json").write_text(json.dumps(colors), encoding="utf-8")
        loader = ThemeLoader(tmp_path)
        theme = loader.load("custom")
        assert theme.name == "custom"
        assert theme.colors["bg"] == "#000000"
        assert "custom" in loader.available()

    def test_missing_keys_error(self, tmp_path):
        (tmp_path / "broken.json").write_text(json.dumps({"bg": "#000000"}), encoding="utf-8")
        loader = ThemeLoader(tmp_path)
        with pytest.raises(ThemeError, match="missing color keys"):
            loader.load("broken")

    def test_invalid_hex_error(self, tmp_path):
        colors = dict(BUILTIN_THEMES["dark"])
        colors["bg"] = "not-a-color"
        (tmp_path / "bad.json").write_text(json.dumps(colors), encoding="utf-8")
        loader = ThemeLoader(tmp_path)
        with pytest.raises(ThemeError, match="hex"):
            loader.load("bad")

    def test_unknown_theme(self, tmp_path):
        loader = ThemeLoader(tmp_path)
        with pytest.raises(ThemeError, match="Unknown theme"):
            loader.load("nope")


class TestAutoTheme:
    def test_detect_dark_by_default(self, monkeypatch):
        monkeypatch.delenv("COLORFGBG", raising=False)
        loader = ThemeLoader()
        assert loader.detect_terminal_background() == "dark"
        assert loader.auto_theme() == "dark"

    def test_detect_light_via_colorfgbg(self, monkeypatch):
        monkeypatch.setenv("COLORFGBG", "15;7")
        loader = ThemeLoader()
        assert loader.detect_terminal_background() == "light"

    def test_detect_dark_via_colorfgbg(self, monkeypatch):
        monkeypatch.setenv("COLORFGBG", "0;0")
        loader = ThemeLoader()
        assert loader.detect_terminal_background() == "dark"

    def test_resolve_auto(self, monkeypatch):
        monkeypatch.setenv("COLORFGBG", "15;7")
        loader = ThemeLoader()
        theme = loader.resolve("auto")
        assert theme.name == "light"
