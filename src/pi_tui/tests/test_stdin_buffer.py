"""StdinBuffer 分片缓冲测试。"""

from __future__ import annotations

from pi_tui.engine import StdinBuffer, is_complete_sequence


def test_partial_csi_waits_for_final_byte() -> None:
    received: list[str] = []
    buffer = StdinBuffer(received.append)
    buffer.process("\x1b[<0;3;5")
    assert received == []
    assert is_complete_sequence("\x1b[<0;3;5") == "incomplete"
    buffer.process("M")
    assert received == ["\x1b[<0;3;5M"]


def test_plain_characters_emit_immediately() -> None:
    received: list[str] = []
    buffer = StdinBuffer(received.append)
    buffer.process("ab")
    assert received == ["a", "b"]


def test_flush_emits_pending() -> None:
    received: list[str] = []
    buffer = StdinBuffer(received.append)
    buffer.process("\x1b[1")
    assert received == []
    assert buffer.flush() == "\x1b[1"
    assert buffer.pending == ""


def test_events_collected_without_callback() -> None:
    buffer = StdinBuffer()
    buffer.process("x\x1b[1")
    assert buffer.events == ["x"]
    buffer.flush()
    assert buffer.events == ["x", "\x1b[1"]
