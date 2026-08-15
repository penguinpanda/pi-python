"""kitty 事件标志（release/repeat）与 OSC 11 通知测试。"""

from __future__ import annotations

import pytest

from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.keys import KeyEvent, KeyParser, parse_input


def test_kitty_repeat_event_parses() -> None:
    # modifier 5 = ctrl+shift；event 2 = repeat。
    events = parse_input(b"\x1b[97;5:2u")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["ctrl+shift+a"]


def test_kitty_release_event_parses_with_release_flag() -> None:
    events = parse_input(b"\x1b[97;5:3u")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert len(keys) == 1
    assert keys[0].name == "ctrl+shift+a"
    assert keys[0].release is True


def test_kitty_release_consumed_incrementally() -> None:
    """release 序列应被增量消费，不残留到后续输入。"""
    parser = KeyParser()
    events = parser.feed(b"\x1b[97;5:3u")
    assert len(events) == 1
    assert events[0].type == "key"
    assert events[0].key is not None and events[0].key.release
    assert not parser.buffer
    events = parser.feed(b"ab")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["a", "b"]


def test_split_csi_sequence_waits_for_completion() -> None:
    """分片 CSI（如方向键）不应被 final flush 丢弃。"""
    parser = KeyParser()
    assert parser.feed(b"\x1b[1;") == []
    assert parser.feed(b"", final=True) == []
    assert parser.buffer
    events = parser.feed(b"5A")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["ctrl+up"]
    assert not parser.buffer


def test_split_osc_sequence_waits_for_completion() -> None:
    parser = KeyParser()
    assert parser.feed(b"\x1b]11;") == []
    assert parser.feed(b"", final=True) == []
    assert parser.buffer
    events = parser.feed(b"#1e1e2e\x07")
    assert [(event.type, event.data) for event in events] == [("osc", "11;#1e1e2e")]
    assert not parser.buffer


def test_esc_control_bytes_normalize_to_named_alt_keys() -> None:
    """回归：ESC+控制字节(alt+enter/alt+backspace)解析为可匹配的键名。"""
    events = parse_input(b"\x1b\r")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["alt+enter"]

    events = parse_input(b"\x1b\x7f")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["alt+backspace"]


def test_esc_plain_char_still_alt_prefixed() -> None:
    """ESC+普通字符仍解析为 alt+<char>(原有行为)。"""
    events = parse_input(b"\x1bx")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["alt+x"]


def test_kitty_modifier_without_flags_still_parses() -> None:
    events = parse_input(b"\x1b[13;4u")
    keys = [event.key for event in events if event.type == "key" and event.key is not None]
    assert [key.name for key in keys] == ["ctrl+enter"]


@pytest.mark.asyncio
async def test_osc11_background_notification() -> None:
    app = App(terminal=FakeTerminal(size=(80, 24)))
    await app._handle_event(KeyEvent(type="osc", data="11;rgb:1e1e/1e2e/1e1e"))
    assert app.osc_background == (30, 30, 30)
    await app._handle_event(KeyEvent(type="osc", data="10;rgb:cdd6/f4"))
    assert app.osc_background == (30, 30, 30)


def test_sgr_mouse_lowercase_m_is_release() -> None:
    events = parse_input(b"\x1b[<0;5;5m")
    mouse = events[0].mouse
    assert mouse is not None
    assert mouse.type == "release"
    assert (mouse.row, mouse.col) == (4, 4)
    events = parse_input(b"\x1b[<0;5;5M")
    assert events[0].mouse is not None
    assert events[0].mouse.type == "press"


def test_focus_events_parse() -> None:
    events = parse_input(b"\x1b[I")
    assert [(event.type, event.data) for event in events] == [("focus", "in")]
    events = parse_input(b"\x1b[O")
    assert [(event.type, event.data) for event in events] == [("focus", "out")]


def test_kitty_flags_response_parses() -> None:
    events = parse_input(b"\x1b[?7u")
    assert [(event.type, event.data) for event in events] == [("kitty_flags", "7")]
