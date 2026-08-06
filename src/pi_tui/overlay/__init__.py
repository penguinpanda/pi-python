"""pi_tui.overlay — overlay 组件树与焦点管理。

核心模型 / 布局 / 焦点状态机不依赖 Textual，可独立单测；
引擎组件（OverlayLayer / OverlayWidget）单独放在 widgets.py。
"""

from .focus import FocusRestoreState, OverlayFocusController
from .layout import OverlayRect, resolve_layout
from .manager import OverlayHooks, OverlayManager
from .model import (
    Margin,
    OverlayBehavior,
    OverlayEntry,
    OverlayHandle,
    OverlayLayout,
    OverlayOptions,
    OverlayStyle,
    RestoreMode,
    parse_overlay_options,
)
from .widgets import OverlayLayer, OverlayWidget

__all__ = [
    "FocusRestoreState",
    "OverlayFocusController",
    "OverlayRect",
    "resolve_layout",
    "OverlayHooks",
    "OverlayManager",
    "Margin",
    "OverlayBehavior",
    "OverlayEntry",
    "OverlayHandle",
    "OverlayLayout",
    "OverlayOptions",
    "OverlayStyle",
    "RestoreMode",
    "parse_overlay_options",
    "OverlayLayer",
    "OverlayWidget",
]
