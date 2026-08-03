"""交互式 TUI 模式（对齐 TS modes/interactive/）。

框架组件位于独立包 pi_tui；本模块提供会话/运行时绑定。
"""

from .app import PiTuiApp, run_tui_mode
from .slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    SlashContext,
    register_builtin_commands,
)

__all__ = [
    "PiTuiApp",
    "run_tui_mode",
    "SlashCommand",
    "SlashCommandRegistry",
    "SlashContext",
    "register_builtin_commands",
]
