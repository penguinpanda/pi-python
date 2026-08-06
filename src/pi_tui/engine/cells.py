"""单元格 / 行模型：组件渲染结果与屏幕差分的基础。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rich.style import Style


@dataclass
class Cell:
    """屏幕单元格：字符 + 样式（Rich Style，保证 Truecolor）。"""

    char: str = " "
    style: Style | None = None
    link: str | None = None


class Line:
    """定宽单元格行，可携带行前 passthrough 协议序列（如 kitty 图像 placement）。"""

    __slots__ = ("cells", "passthrough")

    def __init__(
        self,
        cells: Iterable[Cell] | None = None,
        *,
        passthrough: str = "",
    ) -> None:
        self.cells: list[Cell] = list(cells or [])
        self.passthrough = passthrough

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Line):
            return NotImplemented
        if len(self.cells) != len(other.cells) or self.passthrough != other.passthrough:
            return False
        return all(
            a.char == b.char and a.style == b.style and a.link == b.link
            for a, b in zip(self.cells, other.cells, strict=True)
        )

    def copy(self) -> "Line":
        return Line(
            [Cell(c.char, c.style, c.link) for c in self.cells],
            passthrough=self.passthrough,
        )

    def text(self) -> str:
        return "".join(cell.char for cell in self.cells)

    def patch(self, col: int, other: "Line") -> None:
        """把 other 覆盖到本行 col 处（越界自动裁剪）。"""
        for offset, cell in enumerate(other.cells):
            index = col + offset
            if index < 0:
                continue
            if index >= len(self.cells):
                break
            self.cells[index] = cell

    def signature(self) -> tuple:
        return tuple((cell.char, cell.style, cell.link) for cell in self.cells)


def blank_line(width: int, style: Style | None = None) -> Line:
    return Line([Cell(" ", style) for _ in range(width)])


def line_from_text(text: str, width: int, style: Style | None = None) -> Line:
    """纯文本 → Line（超出宽度截断，不足补空）。"""
    cells = [Cell(char, style) for char in text[:width]]
    cells.extend(Cell(" ", style) for _ in range(max(0, width - len(cells))))
    return Line(cells)


def _style_sgr(style: Style | None) -> str:
    """Rich Style → SGR 前缀。"""
    if style is None:
        return ""
    codes: list[str] = []
    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.underline:
        codes.append("4")
    if style.blink:
        codes.append("5")
    if style.reverse:
        codes.append("7")
    if style.strike:
        codes.append("9")
    if style.color is not None:
        codes.extend(str(code) for code in style.color.get_ansi_codes(foreground=True))
    if style.bgcolor is not None:
        codes.extend(str(code) for code in style.bgcolor.get_ansi_codes(foreground=False))
    return f"\x1b[{';'.join(codes)}m" if codes else ""


def line_to_ansi(line: Line, width: int) -> str:
    """Line → 定宽 ANSI 行（含 OSC 8 链接、尾部重置）。"""
    parts: list[str] = []
    previous: Style | None = None
    previous_link: str | None = None
    for index in range(min(width, len(line.cells))):
        cell = line.cells[index]
        if cell.link != previous_link:
            if cell.link:
                parts.append(f"\x1b]8;{cell.link}\x1b\\")
            else:
                parts.append("\x1b]8;;\x1b\\")
            previous_link = cell.link
        if cell.style != previous:
            sgr = _style_sgr(cell.style)
            if sgr:
                parts.append(sgr)
            previous = cell.style
        parts.append(cell.char)
    remaining = width - min(width, len(line.cells))
    if remaining > 0:
        if previous is not None:
            sgr = _style_sgr(previous)
            if sgr:
                parts.append(sgr)
        parts.append(" " * remaining)
    if previous_link is not None:
        parts.append("\x1b]8;;\x1b\\")
    parts.append("\x1b[0m")
    return "".join(parts)


__all__ = ["Cell", "Line", "blank_line", "line_from_text", "line_to_ansi"]
