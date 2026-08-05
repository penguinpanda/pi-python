"""OSC 11 终端背景查询测试。"""

from __future__ import annotations

import sys

from pi_tui.terminal import (
    _parse_osc_hex_channel,
    parse_osc11_background,
    query_terminal_background,
)


def test_parse_hex6() -> None:
    assert parse_osc11_background("\x1b]11;#1e1e2e\x07") == (30, 30, 46)


def test_parse_hex12() -> None:
    assert parse_osc11_background("\x1b]11;#1e1e1e1e2e2e\x07") == (30, 30, 46)


def test_parse_rgb_forms() -> None:
    assert parse_osc11_background("\x1b]11;rgb:1e1e/1e2e/2e2e\x07") == (30, 30, 46)
    assert parse_osc11_background("\x1b]11;rgb:1e/1e/2e\x1b\\") == (30, 30, 46)
    assert parse_osc11_background("\x1b]11;rgba:ff/ff/ff\x07") == (255, 255, 255)


def test_parse_invalid() -> None:
    assert parse_osc11_background("no response") is None
    assert parse_osc11_background("\x1b]11;#zzzzzz\x07") is None
    assert parse_osc11_background("\x1b]11;rgb:ff/ff\x07") is None


def test_parse_channel_normalization() -> None:
    assert _parse_osc_hex_channel("ff") == 255
    assert _parse_osc_hex_channel("ffff") == 255
    assert _parse_osc_hex_channel("1e") == 30
    assert _parse_osc_hex_channel("1e1e") == 30
    assert _parse_osc_hex_channel("zz") is None


def test_query_non_tty_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert query_terminal_background() is None


def test_query_dumb_term_returns_none(monkeypatch) -> None:
    """TERM=dumb（如 docker exec 无真实终端）时不发查询，避免 OSC 漏到屏幕。"""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "dumb")
    assert query_terminal_background() is None
