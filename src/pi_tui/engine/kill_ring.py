"""Emacs-style kill ring (对齐 TS packages/tui/src/kill-ring.ts)。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KillRing:
    """Ring buffer for kill/yank operations。

    Consecutive kills can accumulate into a single entry; yank-pop cycles
    through older entries via rotate()。
    """

    _ring: list[str] = field(default_factory=list)

    def push(self, text: str, *, prepend: bool, accumulate: bool = False) -> None:
        if not text and not accumulate:
            # Empty entries are preserved for ctrl+k line-join kills.
            self._ring.append("")
            return
        if not text:
            return
        if accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(f"{text}{last}" if prepend else f"{last}{text}")
        else:
            self._ring.append(text)

    def peek(self) -> str | None:
        return self._ring[-1] if self._ring else None

    def rotate(self) -> None:
        if len(self._ring) > 1:
            self._ring.insert(0, self._ring.pop())

    @property
    def entries(self) -> list[str]:
        return list(self._ring)

    def __len__(self) -> int:
        return len(self._ring)

    def __bool__(self) -> bool:
        return bool(self._ring)


__all__ = ["KillRing"]
