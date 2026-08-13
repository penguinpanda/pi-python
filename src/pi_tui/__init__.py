"""pi-tui — 终端 UI 框架（对齐 TS @earendil-works/pi-tui）。

与 pi_ai / pi_agent 平级的可复用框架包：主题、快捷键、基础组件、
选择器与剪贴板图片处理。应用层（AgentSession 绑定）位于
pi_coding_agent.modes.interactive。
"""

from .autocomplete import (
    AutocompleteItem,
    AutocompleteProvider,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
)
from .clipboard_image import ClipboardImage
from .components import MessageEntry, message_to_entries
from .keybindings import (
    DEFAULT_APP_KEYBINDINGS,
    DEFAULT_SESSION_PICKER_KEYBINDINGS,
    Keybinding,
    KeybindingsManager,
)
from .lists import SelectItem, SelectList, SettingItem, SettingsList
from .overlay import (
    Margin,
    OverlayBehavior,
    OverlayEntry,
    OverlayFocusController,
    OverlayHandle,
    OverlayHooks,
    OverlayLayer,
    OverlayLayout,
    OverlayManager,
    OverlayOptions,
    OverlayRect,
    OverlayStyle,
    OverlayWidget,
    RestoreMode,
    FocusRestoreState,
    parse_overlay_options,
    resolve_layout,
)
from .selectors import ModelSelector, SessionPicker
from .terminal import parse_osc11_background, query_terminal_background
from .terminal_image import (
    TerminalImage,
    detect_capabilities,
    encode_iterm2_image,
    encode_kitty_image,
)
from .theme import (
    BUILTIN_THEMES,
    COLOR_KEYS,
    DARK_THEME,
    LIGHT_THEME,
    Theme,
    ThemeError,
    ThemeLoader,
    validate_theme_colors,
)

__all__ = [
    "ClipboardImage",
    "AutocompleteItem",
    "AutocompleteProvider",
    "AutocompleteSuggestions",
    "CombinedAutocompleteProvider",
    "MessageEntry",
    "message_to_entries",
    "DEFAULT_APP_KEYBINDINGS",
    "DEFAULT_SESSION_PICKER_KEYBINDINGS",
    "Keybinding",
    "KeybindingsManager",
    "SelectItem",
    "SelectList",
    "SettingItem",
    "SettingsList",
    "Margin",
    "OverlayBehavior",
    "OverlayEntry",
    "OverlayFocusController",
    "OverlayHandle",
    "OverlayHooks",
    "OverlayLayer",
    "OverlayLayout",
    "OverlayManager",
    "OverlayOptions",
    "OverlayRect",
    "OverlayStyle",
    "OverlayWidget",
    "RestoreMode",
    "FocusRestoreState",
    "parse_overlay_options",
    "resolve_layout",
    "ModelSelector",
    "SessionPicker",
    "parse_osc11_background",
    "query_terminal_background",
    "TerminalImage",
    "detect_capabilities",
    "encode_iterm2_image",
    "encode_kitty_image",
    "BUILTIN_THEMES",
    "COLOR_KEYS",
    "DARK_THEME",
    "LIGHT_THEME",
    "Theme",
    "ThemeError",
    "ThemeLoader",
    "validate_theme_colors",
]
