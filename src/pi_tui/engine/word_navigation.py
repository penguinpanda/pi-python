"""Word navigation primitives (对齐 TS packages/tui/src/word-navigation.ts)。"""

from __future__ import annotations


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def find_word_backward(text: str, cursor: int) -> int:
    """Move one word backward from cursor, skipping trailing separators."""
    cursor = max(0, min(cursor, len(text)))
    index = cursor
    while index > 0 and not _is_word_char(text[index - 1]):
        index -= 1
    while index > 0 and _is_word_char(text[index - 1]):
        index -= 1
    return index


def find_word_forward(text: str, cursor: int, *, include_separator: bool = False) -> int:
    """Move one word forward, optionally including the following separator."""
    cursor = max(0, min(cursor, len(text)))
    index = cursor
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return index
    if _is_word_char(text[index]):
        while index < len(text) and _is_word_char(text[index]):
            index += 1
        if include_separator:
            while index < len(text) and not _is_word_char(text[index]):
                index += 1
        return index
    while index < len(text) and not text[index].isspace() and not _is_word_char(text[index]):
        index += 1
    return index


__all__ = ["find_word_backward", "find_word_forward"]
