"""fuzzy 评分与过滤测试。"""

from __future__ import annotations

from pi_tui.engine import fuzzy_filter, fuzzy_match


def test_exact_match_has_low_score() -> None:
    assert fuzzy_match("abc", "abc").matches
    assert fuzzy_match("abc", "abc").score < -90


def test_subsequence_scores_gap() -> None:
    match = fuzzy_match("abc", "a b c")
    assert match.matches
    assert match.score > fuzzy_match("abc", "abc").score


def test_word_boundary_reward() -> None:
    assert fuzzy_match("b", "beta").score < fuzzy_match("b", "obeta").score


def test_alphanumeric_swap() -> None:
    match = fuzzy_match("m3", "3m")
    assert match.matches


def test_filter_tokens_must_all_match() -> None:
    items = ["foo/bar baz", "foo qux", "bar baz"]
    result = fuzzy_filter(items, "foo baz", str)
    assert result == ["foo/bar baz"]


def test_empty_query_returns_all() -> None:
    assert fuzzy_filter(["a", "b"], "", str) == ["a", "b"]
