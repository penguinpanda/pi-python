"""主题系统（对齐 TS modes/interactive/theme/）。

~40 种命名颜色 + 内置 dark/light + JSON 主题加载 + 终端背景自动检测。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ThemeError(Exception):
    """主题加载/校验错误。"""


# ---------------------------------------------------------------------------
# 命名颜色（对齐 TS theme 的语义键）
# ---------------------------------------------------------------------------

COLOR_KEYS: tuple[str, ...] = (
    # 背景
    "bg",
    "bgAlt",
    "bgBase",
    "bgHover",
    "bgInactive",
    "bgLoading",
    "bgPanel",
    "bgPanelAlt",
    "bgPrompt",
    "bgToolbar",
    "bgUserInput",
    # 边框
    "border",
    "borderActive",
    "borderInactive",
    # 状态
    "error",
    "info",
    "success",
    "warning",
    "accent",
    "accentMuted",
    # 文本
    "text",
    "textAlt",
    "textDim",
    "textDisabled",
    "textLight",
    "textSelected",
    "textSystem",
    "textWarning",
    "dim",
    # 基础色板
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    # Markdown / diff 语义
    "markdownHeading",
    "markdownLink",
    "diffAdd",
    "diffRemove",
    "diffChange",
)


DARK_THEME: dict[str, str] = {
    "bg": "#1e1e2e",
    "bgAlt": "#181825",
    "bgBase": "#11111b",
    "bgHover": "#313244",
    "bgInactive": "#1e1e2e",
    "bgLoading": "#2b2b3a",
    "bgPanel": "#181825",
    "bgPanelAlt": "#1e1e2e",
    "bgPrompt": "#11111b",
    "bgToolbar": "#181825",
    "bgUserInput": "#1e1e2e",
    "border": "#45475a",
    "borderActive": "#89b4fa",
    "borderInactive": "#313244",
    "error": "#f38ba8",
    "info": "#89b4fa",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "accent": "#89b4fa",
    "accentMuted": "#45475a",
    "text": "#cdd6f4",
    "textAlt": "#a6adc8",
    "textDim": "#6c7086",
    "textDisabled": "#45475a",
    "textLight": "#e6e9f0",
    "textSelected": "#11111b",
    "textSystem": "#89b4fa",
    "textWarning": "#f9e2af",
    "dim": "#6c7086",
    "black": "#11111b",
    "red": "#f38ba8",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "blue": "#89b4fa",
    "magenta": "#cba6f7",
    "cyan": "#94e2d5",
    "white": "#cdd6f4",
    "markdownHeading": "#cba6f7",
    "markdownLink": "#89b4fa",
    "diffAdd": "#a6e3a1",
    "diffRemove": "#f38ba8",
    "diffChange": "#f9e2af",
}


LIGHT_THEME: dict[str, str] = {
    "bg": "#eff1f5",
    "bgAlt": "#e6e9ef",
    "bgBase": "#dce0e8",
    "bgHover": "#ccd0da",
    "bgInactive": "#eff1f5",
    "bgLoading": "#e6e9ef",
    "bgPanel": "#e6e9ef",
    "bgPanelAlt": "#eff1f5",
    "bgPrompt": "#dce0e8",
    "bgToolbar": "#e6e9ef",
    "bgUserInput": "#eff1f5",
    "border": "#bcc0cc",
    "borderActive": "#1e66f5",
    "borderInactive": "#ccd0da",
    "error": "#d20f39",
    "info": "#1e66f5",
    "success": "#40a02b",
    "warning": "#df8e1d",
    "accent": "#1e66f5",
    "accentMuted": "#bcc0cc",
    "text": "#4c4f69",
    "textAlt": "#5c5f77",
    "textDim": "#8c8fa1",
    "textDisabled": "#bcc0cc",
    "textLight": "#1e1e2e",
    "textSelected": "#eff1f5",
    "textSystem": "#1e66f5",
    "textWarning": "#df8e1d",
    "dim": "#8c8fa1",
    "black": "#dce0e8",
    "red": "#d20f39",
    "green": "#40a02b",
    "yellow": "#df8e1d",
    "blue": "#1e66f5",
    "magenta": "#8839ef",
    "cyan": "#04a5e5",
    "white": "#4c4f69",
    "markdownHeading": "#8839ef",
    "markdownLink": "#1e66f5",
    "diffAdd": "#40a02b",
    "diffRemove": "#d20f39",
    "diffChange": "#df8e1d",
}


BUILTIN_THEMES: dict[str, dict[str, str]] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Theme:
    """主题快照：名称 + 命名颜色表。"""

    name: str
    colors: dict[str, str]

    def color(self, name: str) -> str:
        return self.colors[name]

    def css_variables(self, prefix: str = "pi") -> dict[str, str]:
        """生成 CSS 变量名 → 色值（供 Textual CSS 模板注入）。"""
        return {f"{prefix}-{key}": value for key, value in self.colors.items()}


# ---------------------------------------------------------------------------
# ThemeLoader
# ---------------------------------------------------------------------------


def validate_theme_colors(colors: dict[str, Any], name: str) -> None:
    """校验主题包含全部命名颜色。"""
    missing = [key for key in COLOR_KEYS if key not in colors]
    if missing:
        raise ThemeError(
            f'Theme "{name}" is missing color keys: {", ".join(missing)}'
        )
    for key, value in colors.items():
        if not isinstance(value, str) or not value.startswith("#"):
            raise ThemeError(
                f'Theme "{name}" color "{key}" must be a hex string, got {value!r}'
            )


class ThemeLoader:
    """加载内置 / 自定义 JSON 主题并自动选择。"""

    def __init__(self, theme_dir: str | Path | None = None) -> None:
        self._theme_dir = Path(theme_dir) if theme_dir else None

    def available(self) -> list[str]:
        names = list(BUILTIN_THEMES)
        if self._theme_dir is not None and self._theme_dir.is_dir():
            names.extend(
                sorted(
                    path.stem
                    for path in self._theme_dir.glob("*.json")
                    if path.is_file()
                )
            )
        return names

    def load(self, name: str) -> Theme:
        """加载主题（内置 dark/light 或 theme_dir 下的 <name>.json）。"""
        if name in BUILTIN_THEMES:
            return Theme(name=name, colors=dict(BUILTIN_THEMES[name]))
        if self._theme_dir is None:
            raise ThemeError(f'Unknown theme: "{name}"')
        path = self._theme_dir / f"{name}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ThemeError(f'Unknown theme: "{name}"') from exc
        except json.JSONDecodeError as exc:
            raise ThemeError(f'Failed to parse theme "{name}": {exc}') from exc
        if not isinstance(raw, dict):
            raise ThemeError(f'Theme "{name}" must be a JSON object')
        validate_theme_colors(raw, name)
        return Theme(name=name, colors=dict(raw))

    def detect_terminal_background(self) -> str:
        """检测终端背景（尽力而为）：dark 或 light。"""
        colorfgbg = os.environ.get("COLORFGBG")
        if colorfgbg:
            parts = colorfgbg.split(";")
            if len(parts) >= 2:
                try:
                    background = int(parts[1])
                    return "light" if background >= 7 else "dark"
                except ValueError:
                    pass
        # Windows 终端默认深色；其余未知时默认 dark。
        return "dark"

    def auto_theme(self) -> str:
        """auto → 根据终端背景选择 dark/light。"""
        return self.detect_terminal_background()

    def resolve(self, name: str | None) -> Theme:
        """解析主题名（None / "auto" 自动选择）。"""
        if not name or name == "auto":
            return self.load(self.auto_theme())
        return self.load(name)


__all__ = [
    "COLOR_KEYS",
    "DARK_THEME",
    "LIGHT_THEME",
    "BUILTIN_THEMES",
    "Theme",
    "ThemeLoader",
    "ThemeError",
    "validate_theme_colors",
]
