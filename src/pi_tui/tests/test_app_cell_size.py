"""App CSI 16t cell size 查询与响应消费。"""

from __future__ import annotations

from pi_tui.engine import FakeTerminal
from pi_tui.engine.app import App


def test_consume_cell_size_response() -> None:
    app = App(terminal=FakeTerminal())
    remaining = app._consume_cell_size_response(b"abc\x1b[6;40;80t")
    assert app.terminal_cell_size == (80, 40)
    assert remaining == b"abc"
    assert app._consume_cell_size_response(b"plain") is None


def test_query_cell_size_requires_image_capability(monkeypatch) -> None:
    app = App(terminal=FakeTerminal())
    terminal = app.terminal
    assert terminal is not None

    monkeypatch.setattr("pi_tui.terminal_image.detect_capabilities", lambda: ())
    app._query_cell_size()
    assert terminal.output == []

    monkeypatch.setattr("pi_tui.terminal_image.detect_capabilities", lambda: ("kitty",))
    app._query_cell_size()
    assert terminal.output == ["\x1b[16t"]
