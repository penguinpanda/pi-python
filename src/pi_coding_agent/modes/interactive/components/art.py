"""图形彩蛋组件（对齐 TS armin/daxnuts/announcement 的静态形态）。"""

from __future__ import annotations

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.widgets import Widget


class ArminComponent(Widget):
    """Armin ASCII 图形，保留 TS 的位图内容。"""

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
        lines: list[Line] = []
        width_cells = min(width, 31)
        for row in range(18):
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


class DaxnutsComponent(Widget):
    """Powered by daxnuts 静态横幅。"""

    def render(self, width: int, height: int) -> list[Line]:
        lines = [
            line_from_text("Powered by daxnuts", width),
            line_from_text("Free Kimi K2.5 via OpenCode Zen", width),
            line_from_text("@thdxr", width),
        ]
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (40, 3)


class EarendilAnnouncementComponent(Widget):
    """Earendil 公告组件。"""

    _BLOG_URL = "https://mariozechner.at/posts/2026-04-08-ive-sold-out/"

    def render(self, width: int, height: int) -> list[Line]:
        lines = [
            line_from_text("pi has joined Earendil", width),
            line_from_text("Read the blog post:", width),
            line_from_text(self._BLOG_URL, width),
        ]
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (60, 3)
