"""pi_tui Terminal / ScreenBuffer / OSC11 补充测试。"""

from __future__ import annotations

import asyncio
import fcntl
import os
import select
import sys

import pytest

from pi_tui.engine.cells import line_from_text
from pi_tui.engine.terminal import ScreenBuffer, Terminal
from pi_tui import terminal as pi_terminal


class _StubOut:
    def __init__(self) -> None:
        self.data: list[str] = []

    def write(self, data: str) -> None:
        self.data.append(data)

    def flush(self) -> None:
        pass


class _StubIn:
    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


class _FakeTtyIn:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0


class _FakeTtyOut:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 1


def test_screen_buffer_diff_and_reset() -> None:
    buffer = ScreenBuffer(width=4, height=2)
    first = buffer.diff([line_from_text("ab", 4), line_from_text("cd", 4)])
    assert "\x1b[2J\x1b[H" in first
    second = buffer.diff([line_from_text("ab", 4), line_from_text("xy", 4)])
    assert "\x1b[2;1H" in second
    buffer.resize(6, 3)
    assert "\x1b[2J\x1b[H" in buffer.diff([line_from_text("abcdef", 6)])
    buffer.reset()
    assert "\x1b[2J\x1b[H" in buffer.diff([line_from_text("z", 6)])


def test_terminal_query_size_fallback(monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "posix")
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        term = Terminal(stdin=_StubIn(fd), stdout=_StubOut())

        def _raise(*args, **kwargs):
            raise OSError("no tty")

        monkeypatch.setattr(fcntl, "ioctl", _raise)
        assert term.query_size() == (80, 24)
        assert term.size == (80, 24)
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_terminal_enter_exit_regular_and_noop(monkeypatch) -> None:
    import termios
    import tty

    monkeypatch.setattr(termios, "tcgetattr", lambda _fd: None)
    monkeypatch.setattr(tty, "setraw", lambda _fd: None)
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        term = Terminal(stdin=_StubIn(fd), stdout=_StubOut(), size=(80, 24))
        await term.enter(alt_screen=False)
        await term.enter(alt_screen=False)
        assert term._entered is True
        assert "".join(term.stdout.data) == ("\x1b[?25l\x1b[?2004h\x1b[>7u\x1b[?u\x1b[c\x1b[?2026h")

        await term.exit(alt_screen=False)
        await term.exit(alt_screen=False)
        assert term._entered is False
        assert "\x1b]133;B\x07" in "".join(term.stdout.data)
    finally:
        os.close(fd)


def test_terminal_cursor_clipboard_and_progress(monkeypatch) -> None:
    out = _StubOut()
    term = Terminal(stdout=out)
    assert term.copy_to_clipboard("hi") is True
    term.set_hardware_cursor(2, 3)
    term.show_cursor()
    term.hide_cursor()
    term.set_color_scheme_notifications(True)
    term.set_progress(True)
    assert "".join(out.data) == ("\x1b]52;c;aGk=\x07\x1b[2;3H\x1b[?25h\x1b[?25l")

    def _fail(_data: str) -> None:
        raise OSError("closed")

    out.write = _fail
    term.copy_to_clipboard("x")


def test_terminal_resize_event(monkeypatch) -> None:
    term = Terminal(size=(80, 24))
    monkeypatch.setattr(term, "query_size", lambda: (100, 30))
    event = term.resize_event()
    assert event is not None
    assert event.type == "resize"
    assert term.resize_event() is None


def test_read_posix_chunk_paths(monkeypatch) -> None:
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        term = Terminal(stdin=_StubIn(fd))

        monkeypatch.setattr(select, "select", lambda *args: ([], [], []))
        assert term._read_posix_chunk() == b""

        def _raise(*args, **kwargs):
            raise OSError("closed")

        monkeypatch.setattr(select, "select", _raise)
        assert term._read_posix_chunk() is None

        monkeypatch.setattr(select, "select", lambda *args: ([fd], [], []))
        monkeypatch.setattr(os, "read", _raise)
        assert term._read_posix_chunk() is None
    finally:
        os.close(fd)


def test_osc11_parse_and_read(monkeypatch) -> None:
    assert pi_terminal._parse_osc_hex_channel("") is None
    assert pi_terminal._parse_osc_hex_channel("zz") is None
    assert pi_terminal.parse_osc11_background("\x1b]11;#1122334\x07") is None

    monkeypatch.setattr(select, "select", lambda *args: ([0], [], []))
    monkeypatch.setattr(os, "read", lambda _fd, _size: b"\x1b]11;#112233\x07")
    assert "#112233" in pi_terminal._read_osc_response_posix(0, 0.01, 0.0)


def test_query_terminal_background_posix(monkeypatch) -> None:
    import termios
    import tty

    monkeypatch.setattr(sys, "stdin", _FakeTtyIn())
    monkeypatch.setattr(sys, "stdout", _FakeTtyOut())
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(termios, "tcgetattr", lambda _fd: None)
    monkeypatch.setattr(tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(termios, "tcsetattr", lambda *args: None)
    monkeypatch.setattr(os, "write", lambda _fd, _data: None)
    monkeypatch.setattr(
        pi_terminal,
        "_read_osc_response_posix",
        lambda _fd, _timeout, _drain: "\x1b]11;#112233\x07",
    )
    assert pi_terminal.query_terminal_background() == (17, 34, 51)

    monkeypatch.setattr(os, "write", lambda _fd, _data: (_ for _ in ()).throw(OSError()))
    assert pi_terminal.query_terminal_background() is None


def test_drain_pending_osc_response(monkeypatch) -> None:
    import termios
    import tty

    monkeypatch.setattr(sys, "stdin", _FakeTtyIn())
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(termios, "tcgetattr", lambda _fd: None)
    monkeypatch.setattr(tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(termios, "tcsetattr", lambda *args: None)
    pi_terminal._OSC11_QUERY_PENDING = True
    pi_terminal.drain_pending_osc_response()
    assert pi_terminal._OSC11_QUERY_PENDING is False


def test_fake_terminal_feed_and_close() -> None:
    from pi_tui.engine.terminal import FakeTerminal

    term = FakeTerminal()
    term.feed_text("hi")
    assert asyncio.run(term.read_chunk()) == b"hi"
    term.close()
    assert asyncio.run(term.read_chunk()) is None
