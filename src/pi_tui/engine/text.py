"""文本渲染：Rich renderable → Line（单元格行）列表。"""

from __future__ import annotations

import re
from typing import Any, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from .cells import Cell, Line

_ANSI_PATTERN = re.compile(
    r"\x1b\[[0-9;:?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][0-9A-Z]"
)


def _make_console(width: int | None = None) -> Console:
    return Console(
        force_terminal=True,
        color_system="truecolor",
        width=width,
        soft_wrap=True,
        file=None,  # type: ignore[arg-type]
    )


def visible_width(text: str) -> int:
    """可见列宽（剥离 ANSI 后按 wcwidth 计算）。"""
    from rich.cells import cell_len

    return cell_len(strip_ansi(text))


def strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列。"""
    return _ANSI_PATTERN.sub("", text)


def _coerce_style(base_style: Style | str | None) -> Style | None:
    if base_style is None:
        return None
    if isinstance(base_style, Style):
        return base_style
    return Style.parse(base_style)


def markup_to_text(markup: str, base_style: Style | str | None = None) -> Text:
    """Rich markup 字符串 → Text。"""
    style = _coerce_style(base_style)
    try:
        if style is None:
            return Text.from_markup(markup)
        return Text.from_markup(markup, style=style)
    except Exception:
        if style is None:
            return Text(markup)
        return Text(markup, style=style)


def segments_to_line(segments: list[Segment], width: int) -> Line:
    """Rich Segment 列表 → 定宽 Line（含 OSC8 链接）。"""
    cells: list[Cell] = []
    for segment in segments:
        text = segment.text
        if not text:
            continue
        link = None
        if segment.style is not None:
            link_value = getattr(segment.style, "link", None)
            if isinstance(link_value, str) and link_value:
                link = link_value
        for char in text:
            if len(cells) >= width:
                break
            cells.append(Cell(char, segment.style, link))
    while len(cells) < width:
        style = cells[-1].style if cells else None
        cells.append(Cell(" ", style))
    return Line(cells)


def render_renderable(renderable: Any, width: int) -> list[Line]:
    """任意 Rich renderable → 定宽 Line 列表。"""
    console = _make_console(width)
    options = console.options
    options.update(width=width)
    lines: list[list[Segment]] = console.render_lines(renderable, options)
    return [segments_to_line(line, width) for line in lines]


def render_markup(markup: str, width: int, base_style: Style | str | None = None) -> list[Line]:
    """Rich markup 字符串 → 定宽 Line 列表。"""
    return render_renderable(markup_to_text(markup, base_style), width)


def render_markdown(
    markdown_text: str,
    width: int,
    *,
    code_theme: str = "monokai",
    theme_colors: dict | None = None,
) -> list[Line]:
    """Markdown → 定宽 Line 列表；theme_colors 时按主题重着色标题/代码。"""
    lines = render_renderable(Markdown(markdown_text, code_theme=code_theme), width)
    if theme_colors:
        _apply_markdown_theme(lines, theme_colors)
    return lines


def _apply_markdown_theme(lines: list[Line], theme_colors: dict) -> None:
    """Rich Markdown 输出 → pi 主题色（标题=accent，代码=面板底色+文本色）。"""
    from rich.color import Color

    def _color(value):
        if isinstance(value, tuple):
            return Color.from_rgb(*value)
        return value

    heading = _color(theme_colors.get("heading"))
    code_fg = _color(theme_colors.get("code_fg"))
    code_bg = _color(theme_colors.get("code_bg"))
    for line in lines:
        for cell in line.cells:
            style = cell.style
            if style is None:
                continue
            if style.bold and style.underline and heading is not None:
                cell.style = Style(bold=True, color=heading)
            elif style.bgcolor is not None and (code_fg is not None or code_bg is not None):
                cell.style = style + Style(color=code_fg, bgcolor=code_bg)


def render_group(items: Sequence[Any], width: int) -> list[Line]:
    """多个 renderable 垂直拼接渲染。"""
    from rich.console import Group

    return render_renderable(Group(*items), width)


__all__ = [
    "markup_to_text",
    "render_markdown",
    "render_markup",
    "render_renderable",
    "render_group",
    "strip_ansi",
    "visible_width",
]
