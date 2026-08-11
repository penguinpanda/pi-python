"""Terminal Protocol 抽象测试。"""

from __future__ import annotations

from pi_tui.engine import FakeTerminal, ProcessTerminal, Terminal, TerminalProtocol


def test_terminal_implements_protocol() -> None:
    assert isinstance(Terminal(), TerminalProtocol)


def test_fake_terminal_implements_protocol() -> None:
    assert isinstance(FakeTerminal(), TerminalProtocol)


def test_process_terminal_alias() -> None:
    assert ProcessTerminal is Terminal
