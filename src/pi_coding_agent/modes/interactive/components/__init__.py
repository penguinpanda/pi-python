"""coding-agent 交互模式专用组件。"""

from .art import ArminComponent, DaxnutsComponent, EarendilAnnouncementComponent
from .config_selector import (
    ConfigScope,
    ConfigSelectorComponent,
    ConfigSelectorModel,
    ResourceGroup,
    ResourceItem,
    ResourceOrigin,
    ResourceScope,
    ResourceType,
)
from .dialogs import (
    LoginDialogComponent,
    ShowImagesSelectorComponent,
)
from .diff import DiffEntry, render_diff_lines
from pi_tui.selectors import (
    ExtensionSelector as ExtensionSelectorComponent,
    ModelSelector as ModelSelectorComponent,
    OAuthSelector as OAuthSelectorComponent,
    ScopedModelsSelector as ScopedModelsSelectorComponent,
    SessionPicker as SessionSelectorComponent,
    SettingsSelector as SettingsSelectorComponent,
    ThinkingSelector as ThinkingSelectorComponent,
    TreeSelector as TreeSelectorComponent,
    TrustSelector as TrustSelectorComponent,
)
from .first_time_setup import FirstTimeSetupComponent
from .basic import (
    PiChatContainer,
    PiEditor,
    PiEditorVim,
    PiFooter,
    PiHeader,
    PiStatusBar,
    PiToolbar,
)

renderDiff = render_diff_lines

__all__ = [
    "ArminComponent",
    "DaxnutsComponent",
    "EarendilAnnouncementComponent",
    "ConfigSelectorComponent",
    "ConfigSelectorModel",
    "ConfigScope",
    "ResourceGroup",
    "ResourceItem",
    "ResourceOrigin",
    "ResourceScope",
    "ResourceType",
    "LoginDialogComponent",
    "ShowImagesSelectorComponent",
    "DiffEntry",
    "FirstTimeSetupComponent",
    "PiChatContainer",
    "PiEditor",
    "PiEditorVim",
    "PiFooter",
    "PiHeader",
    "PiStatusBar",
    "PiToolbar",
    "render_diff_lines",
    "renderDiff",
    "ModelSelectorComponent",
    "SessionSelectorComponent",
    "TreeSelectorComponent",
    "SettingsSelectorComponent",
    "ThinkingSelectorComponent",
    "OAuthSelectorComponent",
    "ScopedModelsSelectorComponent",
    "ExtensionSelectorComponent",
    "TrustSelectorComponent",
]
