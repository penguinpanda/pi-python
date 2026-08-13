"""图形彩蛋组件（对齐 TS armin/daxnuts/announcement）。"""

from __future__ import annotations

from pathlib import Path

from rich.color import Color
from rich.style import Style

from pi_tui.engine.cells import Cell, Line, blank_line, line_from_text
from pi_tui.engine.widgets import Widget
from pi_tui.terminal_image import TerminalImage

from ._daxnuts import DAX_HEX


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def _pixels_from_hex(
    hex_data: str, width: int = 32, height: int = 32
) -> list[list[tuple[int, int, int]]]:
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(height):
        row: list[tuple[int, int, int]] = []
        for x in range(width):
            index = (y * width + x) * 6
            chunk = hex_data[index : index + 6]
            row.append(
                (
                    int(chunk[0:2], 16),
                    int(chunk[2:4], 16),
                    int(chunk[4:6], 16),
                )
            )
        rows.append(row)
    return rows


def _dax_image_lines(width: int) -> list[Line]:
    pixels = _pixels_from_hex(DAX_HEX)
    lines: list[Line] = []
    for y in range(0, 32, 2):
        cells: list[Cell] = []
        for x in range(32):
            top = pixels[y][x]
            bottom = pixels[y + 1][x]
            style = Style(
                color=Color.from_rgb(*bottom),
                bgcolor=Color.from_rgb(*top),
            )
            cells.append(Cell("▄", style))
        line = blank_line(width)
        line.patch(0, Line(cells))
        lines.append(line)
    return lines


class _AnimatedArt(Widget):
    """带一次性渐进动画的图形组件。"""

    frame_count = 24
    delay_seconds = 0.08

    def __init__(self) -> None:
        super().__init__()
        self._frame = self.frame_count
        self._scheduled = False

    def _ensure_animation(self) -> None:
        if self._scheduled or self.app is None:
            return
        self._scheduled = True
        self._frame = 0

        import asyncio

        asyncio.get_event_loop().call_later(self.delay_seconds, self._advance)

    def _advance(self) -> None:
        self._frame = min(self.frame_count, self._frame + 1)
        self.refresh()
        if self._frame < self.frame_count:
            import asyncio

            asyncio.get_event_loop().call_later(self.delay_seconds, self._advance)


class ArminComponent(_AnimatedArt):
    """Armin ASCII 图形（逐行扫描揭示）。"""

    _BITS = [
        0xFF,
        0xFF,
        0xFF,
        0x7F,
        0xFF,
        0xF0,
        0xFF,
        0x7F,
        0xFF,
        0xED,
        0xFF,
        0x7F,
        0xFF,
        0xDB,
        0xFF,
        0x7F,
        0xFF,
        0xB7,
        0xFF,
        0x7F,
        0xFF,
        0x77,
        0xFE,
        0x7F,
        0x3F,
        0xF8,
        0xFE,
        0x7F,
        0xDF,
        0xFF,
        0xFE,
        0x7F,
        0xDF,
        0x3F,
        0xFC,
        0x7F,
        0x9F,
        0xC3,
        0xFB,
        0x7F,
        0x6F,
        0xFC,
        0xF4,
        0x7F,
        0xF7,
        0x0F,
        0xF7,
        0x7F,
        0xF7,
        0xFF,
        0xF7,
        0x7F,
        0xF7,
        0xFF,
        0xE3,
        0x7F,
        0xF7,
        0x07,
        0xE8,
        0x7F,
        0xEF,
        0xF8,
        0x67,
        0x70,
        0x0F,
        0xFF,
        0xBB,
        0x6F,
        0xF1,
        0x00,
        0xD0,
        0x5B,
        0xFD,
        0x3F,
        0xEC,
        0x53,
        0xC1,
        0xFF,
        0xEF,
        0x57,
        0x9F,
        0xFD,
        0xEE,
        0x5F,
        0x9F,
        0xFC,
        0xAE,
        0x5F,
        0x1F,
        0x78,
        0xAC,
        0x5F,
        0x3F,
        0x00,
        0x50,
        0x6C,
        0x7F,
        0x00,
        0xDC,
        0x77,
        0xFF,
        0xC0,
        0x3F,
        0x78,
        0xFF,
        0x01,
        0xF8,
        0x7F,
        0xFF,
        0x03,
        0x9C,
        0x78,
        0xFF,
        0x07,
        0x8C,
        0x7C,
        0xFF,
        0x0F,
        0xCE,
        0x78,
        0xFF,
        0xFF,
        0xCF,
        0x7F,
        0xFF,
        0xFF,
        0xCF,
        0x78,
        0xFF,
        0xFF,
        0xDF,
        0x78,
        0xFF,
        0xFF,
        0xDF,
        0x7D,
        0xFF,
        0xFF,
        0x3F,
        0x7E,
        0xFF,
        0xFF,
        0xFF,
        0x7F,
    ]

    def render(self, width: int, height: int) -> list[Line]:
        self._ensure_animation()
        lines: list[Line] = []
        width_cells = min(width, 31)
        visible_rows = round(18 * (self._frame / max(1, self.frame_count)))
        for row in range(18):
            if row >= visible_rows:
                lines.append(blank_line(width))
                continue
            chars: list[str] = []
            for x in range(width_cells):
                upper = self._pixel(x, row * 2)
                lower = self._pixel(x, row * 2 + 1)
                if upper and lower:
                    chars.append("█")
                elif upper:
                    chars.append("▀")
                elif lower:
                    chars.append("▄")
                else:
                    chars.append(" ")
            lines.append(line_from_text(" " + "".join(chars), width))
        if self._frame >= self.frame_count:
            lines.append(line_from_text(" ARMIN SAYS HI", width))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def _pixel(self, x: int, y: int) -> bool:
        if y >= 36:
            return False
        byte_index = y * 4 + x // 8
        if byte_index >= len(self._BITS):
            return False
        return ((self._BITS[byte_index] >> (x % 8)) & 1) == 0

    def content_size(self) -> tuple[int, int]:
        return (32, 19)


class DaxnutsComponent(_AnimatedArt):
    """Powered by daxnuts RGB 半块图像与说明文字。"""

    def render(self, width: int, height: int) -> list[Line]:
        self._ensure_animation()
        image_lines = _dax_image_lines(width)
        reveal = round(len(image_lines) * (self._frame / max(1, self.frame_count)))
        lines: list[Line] = [blank_line(width)]
        for index, image_line in enumerate(image_lines):
            lines.append(image_line if index < reveal else blank_line(width))
        lines.append(blank_line(width))
        if self._frame >= self.frame_count:
            lines.extend(
                [
                    line_from_text("Free Kimi K2.5 via OpenCode Zen", width),
                    line_from_text('"Powered by daxnuts"', width),
                    line_from_text("— @thdxr", width),
                ]
            )
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (32, 21)


class EarendilAnnouncementComponent(Widget):
    """Earendil 公告组件。"""

    _BLOG_URL = "https://mariozechner.at/posts/2026-04-08-ive-sold-out/"

    def render(self, width: int, height: int) -> list[Line]:
        lines = [
            line_from_text("pi has joined Earendil", width),
            line_from_text("Read the blog post:", width),
            line_from_text(self._BLOG_URL, width),
            blank_line(width),
        ]
        image_path = _ASSET_DIR / "clankolas.png"
        if image_path.is_file():
            lines.extend(TerminalImage(str(image_path), name="clankolas.png").render(width, 1))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (60, 5)
