"""Fuzzy matching and filtering (对齐 TS packages/tui/src/fuzzy.ts)。"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FuzzyMatch:
    matches: bool
    score: float


def _match_query(normalized_query: str, text_lower: str) -> FuzzyMatch:
    if not normalized_query:
        return FuzzyMatch(True, 0.0)
    if len(normalized_query) > len(text_lower):
        return FuzzyMatch(False, 0.0)

    query_index = 0
    score = 0.0
    last_match_index = -1
    consecutive_matches = 0
    for index, char in enumerate(text_lower):
        if query_index >= len(normalized_query):
            break
        if char != normalized_query[query_index]:
            continue
        is_word_boundary = index == 0 or text_lower[index - 1] in " \t-_./:"
        if last_match_index == index - 1:
            consecutive_matches += 1
            score -= consecutive_matches * 5
        else:
            consecutive_matches = 0
            if last_match_index >= 0:
                score += (index - last_match_index - 1) * 2
        if is_word_boundary:
            score -= 10
        score += index * 0.1
        last_match_index = index
        query_index += 1

    if query_index < len(normalized_query):
        return FuzzyMatch(False, 0.0)
    if normalized_query == text_lower:
        score -= 100
    return FuzzyMatch(True, score)


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    """Lower score is better; empty query matches everything."""
    query_lower = query.lower()
    text_lower = text.lower()
    primary = _match_query(query_lower, text_lower)
    if primary.matches:
        return primary

    alpha_numeric = re.fullmatch(r"([a-z]+)([0-9]+)", query_lower)
    numeric_alpha = re.fullmatch(r"([0-9]+)([a-z]+)", query_lower)
    if alpha_numeric is not None:
        swapped = f"{alpha_numeric.group(2)}{alpha_numeric.group(1)}"
    elif numeric_alpha is not None:
        swapped = f"{numeric_alpha.group(2)}{numeric_alpha.group(1)}"
    else:
        return primary

    swapped_match = _match_query(swapped, text_lower)
    if not swapped_match.matches:
        return primary
    return FuzzyMatch(True, swapped_match.score + 5)


def fuzzy_filter(items: Sequence[T], query: str, get_text: Callable[[T], str]) -> list[T]:
    """Filter and sort by fuzzy quality; whitespace/slash tokens all must match."""
    if not query.strip():
        return list(items)
    tokens = [token for token in re.split(r"[\s/]+", query.strip()) if token]
    if not tokens:
        return list(items)

    scored: list[tuple[float, T]] = []
    for item in items:
        text = get_text(item)
        total = 0.0
        ok = True
        for token in tokens:
            match = fuzzy_match(token, text)
            if match.matches:
                total += match.score
            else:
                ok = False
                break
        if ok:
            scored.append((total, item))
    scored.sort(key=lambda pair: pair[0])
    return [item for _score, item in scored]


__all__ = ["FuzzyMatch", "fuzzy_match", "fuzzy_filter"]
