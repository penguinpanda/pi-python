"""Diff 渲染（对齐 TS components/diff.ts）。"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

from rich.style import Style

from pi_tui.engine.cells import Cell, Line, blank_line, line_from_text
from pi_tui.engine.widgets import Widget


_DIFF_LINE_RE = re.compile(r"^([+\- ])(\s*\d*)\s(.*)$")


def _parse_diff_line(line: str) -> tuple[str, str, str] | None:
    match = _DIFF_LINE_RE.match(line)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def _changes_from_matcher(
    old_tokens: list[str],
    new_tokens: list[str],
) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    matcher = difflib.SequenceMatcher(
        a=old_tokens,
        b=new_tokens,
        autojunk=False,
    )
    removed: list[tuple[str, bool]] = []
    added: list[tuple[str, bool]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        old_part = " ".join(old_tokens[i1:i2])
        new_part = " ".join(new_tokens[j1:j2])
        if op == "equal":
            removed.append((old_part, False))
            added.append((new_part, False))
        elif op == "replace":
            removed.append((old_part, True))
            added.append((new_part, True))
        elif op == "delete":
            removed.append((old_part, True))
        elif op == "insert":
            added.append((new_part, True))
    return removed, added


def _word_changes(old: str, new: str) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """按词级差异着色；无有效词边界时回退到逐字符。"""
    removed, added = _changes_from_matcher(old.split(), new.split())
    if not any(changed for _text, changed in removed + added):
        removed, added = _changes_from_matcher(list(old), list(new))
    return removed, added


def _line_from_parts(
    prefix: str,
    parts: Iterable[tuple[str, bool]],
    width: int,
    base: Style,
) -> Line:
    cells: list[Cell] = [Cell(char, base) for char in prefix]
    for text, changed in parts:
        style = base + Style(reverse=True) if changed else base
        cells.extend(Cell(char, style) for char in text)
    while len(cells) < width:
        cells.append(Cell(" ", base))
    return Line(cells[:width])


def render_diff_lines(
    diff_text: str,
    width: int,
    colors: dict[str, str] | None = None,
) -> list[Line]:
    colors = colors or {}
    context_style = Style(color=colors.get("toolDiffContext", colors.get("dim")))
    removed_style = Style(color=colors.get("toolDiffRemoved", colors.get("diffRemove")))
    added_style = Style(color=colors.get("toolDiffAdded", colors.get("diffAdd")))
    lines: list[Line] = []
    raw_lines = diff_text.splitlines() or [""]
    index = 0
    while index < len(raw_lines):
        parsed = _parse_diff_line(raw_lines[index])
        if parsed is None:
            lines.append(line_from_text(raw_lines[index], width, context_style))
            index += 1
            continue
        prefix, line_num, content = parsed
        if prefix == "-":
            removed: list[tuple[str, str, str]] = []
            while index < len(raw_lines):
                current = _parse_diff_line(raw_lines[index])
                if current is None or current[0] != "-":
                    break
                removed.append(current)
                index += 1
            added: list[tuple[str, str, str]] = []
            while index < len(raw_lines):
                current = _parse_diff_line(raw_lines[index])
                if current is None or current[0] != "+":
                    break
                added.append(current)
                index += 1
            if len(removed) == 1 and len(added) == 1:
                old_parts, new_parts = _word_changes(removed[0][2], added[0][2])
                lines.append(
                    _line_from_parts(
                        f"-{removed[0][1]} ",
                        old_parts,
                        width,
                        removed_style,
                    )
                )
                lines.append(
                    _line_from_parts(
                        f"+{added[0][1]} ",
                        new_parts,
                        width,
                        added_style,
                    )
                )
            else:
                for _prefix, num, text in removed:
                    lines.append(
                        line_from_text(f"-{num} {_replace_tabs(text)}", width, removed_style)
                    )
                for _prefix, num, text in added:
                    lines.append(
                        line_from_text(f"+{num} {_replace_tabs(text)}", width, added_style)
                    )
            continue
        if prefix == "+":
            lines.append(
                line_from_text(f"+{line_num} {_replace_tabs(content)}", width, added_style)
            )
        else:
            lines.append(
                line_from_text(f" {line_num} {_replace_tabs(content)}", width, context_style)
            )
        index += 1
    return lines


class DiffEntry(Widget):
    """把 diff 文本渲染为聊天条目。"""

    def __init__(self, diff_text: str, *, theme_colors: dict[str, str] | None = None) -> None:
        super().__init__()
        self.diff_text = diff_text
        self.theme_colors = dict(theme_colors or {})

    def render(self, width: int, height: int) -> list[Line]:
        lines = render_diff_lines(self.diff_text, max(0, width - 2), self.theme_colors)
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (80, min(1 + len(self.diff_text.splitlines()), 20))
