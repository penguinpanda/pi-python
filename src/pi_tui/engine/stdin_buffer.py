"""Stdin sequence buffer (对齐 TS packages/tui/src/stdin-buffer.ts)。"""

from __future__ import annotations

import asyncio

from typing import Any, Callable

ESC = "\x1b"


def _is_complete_csi(data: str) -> bool:
    payload = data[2:]
    for char in payload:
        if 0x40 <= ord(char) <= 0x7E:
            if payload.startswith("<") and char in ("M", "m"):
                parts = payload[1:-1].split(";")
                return len(parts) == 3 and all(part.isdigit() for part in parts)
            return True
    return False


def _is_complete_osc(data: str) -> bool:
    return data.endswith("\x07") or data.endswith(f"{ESC}\\")


def is_complete_sequence(data: str) -> str:
    """Classify a buffer prefix as complete/incomplete/not-escape."""
    if not data.startswith(ESC):
        return "not-escape"
    if len(data) == 1:
        return "incomplete"
    after_esc = data[1:]
    if after_esc.startswith("["):
        return "complete" if _is_complete_csi(data) else "incomplete"
    if after_esc.startswith("]"):
        return "complete" if _is_complete_osc(data) else "incomplete"
    if after_esc.startswith(("P", "_")):
        return "complete" if data.endswith(f"{ESC}\\") else "incomplete"
    if after_esc.startswith("O"):
        return "complete" if len(after_esc) >= 2 else "incomplete"
    return "complete"


def extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    """Split complete escape sequences and plain characters from a buffer."""
    sequences: list[str] = []
    pos = 0
    while pos < len(buffer):
        remaining = buffer[pos:]
        if not remaining.startswith(ESC):
            sequences.append(remaining[0])
            pos += 1
            continue
        seq_end = 1
        while seq_end <= len(remaining):
            candidate = remaining[:seq_end]
            status = is_complete_sequence(candidate)
            if status == "complete":
                sequences.append(candidate)
                pos += seq_end
                break
            if status == "not-escape":
                sequences.append(candidate)
                pos += seq_end
                break
            seq_end += 1
        else:
            return sequences, remaining
    return sequences, ""


class StdinBuffer:
    """Buffers partial input and emits complete sequences."""

    def __init__(
        self,
        on_data: Callable[[str], None] | None = None,
        *,
        timeout: float = 0.01,
    ) -> None:
        self.on_data = on_data
        self._buffer = ""
        self._timeout = max(0.0, timeout)
        self._timer: Any = None
        self.events: list[str] = []

    def process(self, data: str) -> None:
        self._buffer += data
        self._flush_complete()
        self._schedule_flush()

    def flush(self) -> str | None:
        self._cancel_timer()
        if not self._buffer:
            return None
        data, self._buffer = self._buffer, ""
        self._emit(data)
        return data

    def cancel(self) -> None:
        self._cancel_timer()
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    def _flush_complete(self) -> None:
        sequences, remainder = extract_complete_sequences(self._buffer)
        self._buffer = remainder
        for sequence in sequences:
            self._emit(sequence)

    def _emit(self, data: str) -> None:
        if self.on_data is not None:
            self.on_data(data)
        else:
            self.events.append(data)

    def _schedule_flush(self) -> None:
        if not self._buffer:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cancel_timer()
        self._timer = loop.call_later(self._timeout, self._on_timeout)

    def _on_timeout(self) -> None:
        self._timer = None
        if self._buffer:
            data, self._buffer = self._buffer, ""
            self._emit(data)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


__all__ = ["StdinBuffer", "is_complete_sequence", "extract_complete_sequences"]
