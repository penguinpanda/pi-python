"""Editor component interface (对齐 TS packages/tui/src/editor-component.ts)。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .keys import Key


@runtime_checkable
class EditorComponent(Protocol):
    """Custom editor contract used by the TUI application."""

    def get_text(self) -> str: ...

    def set_text(self, text: str) -> None: ...

    def handle_key(self, key: Key) -> bool: ...

    def add_to_history(self, text: str) -> None: ...

    def insert_text_at_cursor(self, text: str) -> None: ...

    def get_expanded_text(self) -> str: ...

    def set_autocomplete_provider(self, provider: Any) -> None: ...


__all__ = ["EditorComponent"]
