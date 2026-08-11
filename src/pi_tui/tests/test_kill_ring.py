"""KillRing 独立类测试。"""

from __future__ import annotations

from pi_tui.engine import KillRing


def test_push_peek_and_len() -> None:
    ring = KillRing()
    ring.push("a", prepend=False)
    ring.push("b", prepend=False)
    assert len(ring) == 2
    assert ring.peek() == "b"
    assert ring.entries == ["a", "b"]


def test_accumulate_appends_and_prepends() -> None:
    ring = KillRing()
    ring.push("ab", prepend=False)
    ring.push("cd", prepend=False, accumulate=True)
    assert ring.peek() == "abcd"
    ring.push("xy", prepend=True, accumulate=True)
    assert ring.peek() == "xyabcd"


def test_rotate_cycles_for_yank_pop() -> None:
    ring = KillRing()
    ring.push("a", prepend=False)
    ring.push("b", prepend=False)
    ring.rotate()
    assert ring.peek() == "a"
    ring.rotate()
    assert ring.peek() == "b"


def test_empty_entry_preserved_for_line_join() -> None:
    ring = KillRing()
    ring.push("", prepend=False)
    assert ring.entries == [""]
    assert ring.peek() == ""
