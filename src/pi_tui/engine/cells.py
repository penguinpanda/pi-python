"""单元格 / 行模型：组件渲染结果与屏幕差分的基础。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rich.cells import cell_len
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


def _visible_slice(cells: Iterable[Cell], width: int) -> tuple[list[Cell], int]:
    """按终端可见列宽截取单元格（CJK 等宽字符不拆半、不超宽）。"""
    result: list[Cell] = []
    used = 0
    for cell in cells:
        char_width = cell_len(cell.char)
        if used + char_width > width:
            break
        result.append(cell)
        used += char_width
    return result, used


def line_from_text(text: str, width: int, style: Style | None = None) -> Line:
    """纯文本 → Line（按可见列宽截断，不足补空；CJK 按 2 列计）。"""
    cells, used = _visible_slice((Cell(char, style) for char in text), width)
    cells.extend(Cell(" ", style) for _ in range(max(0, width - used)))
    return Line(cells)


def _style_sgr(style: Style | None, previous: Style | None = None) -> str:
    """Rich Style → SGR 前缀；previous 提供时补发属性关闭码（如 27 关闭反色）。

    只发开启码会让被移除的属性（bold/underline/reverse 等）残留到整行，
    例如光标格反色后下一格未发 \x1b[27m，导致整行都被反色覆盖。
    """
    if style is None:
        return "\x1b[0m" if previous is not None else ""
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
        codes.extend(style.color.get_ansi_codes(foreground=True))
    if style.bgcolor is not None:
        codes.extend(style.bgcolor.get_ansi_codes(foreground=False))
    if previous is not None:
        if not style.bold and previous.bold:
            codes.append("22")
        if not style.dim and previous.dim and "22" not in codes:
            codes.append("22")
        if not style.italic and previous.italic:
            codes.append("23")
        if not style.underline and previous.underline:
            codes.append("24")
        if not style.blink and previous.blink:
            codes.append("25")
        if not style.reverse and previous.reverse:
            codes.append("27")
        if not style.strike and previous.strike:
            codes.append("29")
        if style.color is None and previous.color is not None:
            codes.append("39")
        if style.bgcolor is None and previous.bgcolor is not None:
            codes.append("49")
    return f"\x1b[{';'.join(codes)}m" if codes else ""


def line_to_ansi(line: Line, width: int) -> str:
    """Line → 定宽 ANSI 行（含 OSC 8 链接、尾部重置）。"""
    parts: list[str] = []
    previous: Style | None = None
    previous_link: str | None = None
    cells, used = _visible_slice(line.cells, width)
    for cell in cells:
        if cell.link != previous_link:
            if cell.link:
                parts.append(f"\x1b]8;{cell.link}\x1b\\")
            else:
                parts.append("\x1b]8;;\x1b\\")
            previous_link = cell.link
        if cell.style != previous:
            sgr = _style_sgr(cell.style, previous)
            if sgr:
                parts.append(sgr)
            previous = cell.style
        parts.append(cell.char)
    remaining = width - used
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
