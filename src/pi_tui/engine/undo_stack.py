"""Generic undo stack with clone-on-push (对齐 TS packages/tui/src/undo-stack.ts)。"""

from __future__ import annotations

import copy

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class UndoStack(Generic[T]):
    """Stores deep clones of state snapshots."""

    _stack: list[T] = field(default_factory=list)
    max_length: int | None = None

    def push(self, state: T) -> None:
        self._stack.append(copy.deepcopy(state))
        if self.max_length is not None:
            while len(self._stack) > self.max_length:
                self._stack.pop(0)

    def pop(self) -> T | None:
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        self._stack.clear()

    def __len__(self) -> int:
        return len(self._stack)

    def __bool__(self) -> bool:
        return bool(self._stack)


__all__ = ["UndoStack"]
