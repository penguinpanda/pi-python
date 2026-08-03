"""扩展系统（Phase 5）——对齐 TS core/extensions/。"""

from .loader import ExtensionLoader, LoadExtensionsResult
from .registry import ExtensionRegistry
from .runner import ExtensionCommandContext, ExtensionContext, ExtensionRunner
from .types import (
    EventBus,
    Extension,
    ExtensionAPI,
    ExtensionError,
    ExtensionFlag,
    ExtensionRuntime,
    ExtensionShortcut,
    NoopUIContext,
    RegisteredCommand,
    ToolDefinition,
    UIContext,
)

__all__ = [
    "ExtensionLoader",
    "LoadExtensionsResult",
    "ExtensionRegistry",
    "ExtensionRunner",
    "ExtensionContext",
    "ExtensionCommandContext",
    "EventBus",
    "Extension",
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionFlag",
    "ExtensionRuntime",
    "ExtensionShortcut",
    "NoopUIContext",
    "RegisteredCommand",
    "ToolDefinition",
    "UIContext",
]
