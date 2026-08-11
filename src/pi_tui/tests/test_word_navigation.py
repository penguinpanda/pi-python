"""Unicode 单词导航测试。"""

from __future__ import annotations

from pi_tui.engine import find_word_backward, find_word_forward


def test_backward_skips_spaces_and_stops_at_word() -> None:
    assert find_word_backward("one two three", 10) == 8
    assert find_word_backward("one two three", 8) == 4
    assert find_word_backward("one two three", 0) == 0


def test_forward_skips_leading_spaces() -> None:
    assert find_word_forward("one two three", 0) == 3
    assert find_word_forward("one two three", 4) == 7
    assert find_word_forward("one two three", 11) == 13


def test_punctuation_boundary() -> None:
    assert find_word_backward("foo.bar", 7) == 4
    assert find_word_forward("foo.bar", 0) == 3
    assert find_word_forward("foo.bar", 0, include_separator=True) == 4


def test_cjk_and_emoji() -> None:
    assert find_word_backward("你好 world", 3) == 0
    assert find_word_forward("你好 world", 0) == 2
    assert find_word_forward("hello 😀", 5) == 7
