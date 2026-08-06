"""终端图像协议（kitty / iTerm2）序列生成，独立可单测。"""

from __future__ import annotations

import base64
import os

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.widgets import Widget


def detect_capabilities() -> tuple[str, ...]:
    """尽力探测终端图像能力（kitty / iTerm2）。"""
    capabilities: list[str] = []
    term = os.environ.get("TERM", "").lower()
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if "kitty" in term or program == "kitty":
        capabilities.append("kitty")
    if "iterm" in program:
        capabilities.append("iterm2")
    return tuple(capabilities)


def _encode_chunks(header: str, data: bytes, chunk_size: int) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    parts: list[str] = []
    for index in range(0, len(encoded), chunk_size):
        chunk = encoded[index : index + chunk_size]
        more = "0" if index + chunk_size >= len(encoded) else "1"
        parts.append(f"\x1b_G{header},m={more};{chunk}\x1b\\")
    return "".join(parts)


def encode_kitty_image(
    data: bytes,
    *,
    width: int | None = None,
    height: int | None = None,
    chunk_size: int = 4096,
) -> str:
    """把图片编码为 kitty graphics protocol 传输序列（a=T）。"""
    controls = ["a=T", "f=100"]
    if width is not None:
        controls.append(f"s={int(width)}")
    if height is not None:
        controls.append(f"v={int(height)}")
    return _encode_chunks(",".join(controls), data, chunk_size)


def encode_kitty_placement(
    data: bytes,
    *,
    image_id: int = 0,
    width: int | None = None,
    height: int | None = None,
    chunk_size: int = 4096,
) -> str:
    """把图片编码为 kitty placement 序列（a=p），在当前光标处显示。

    `image_id` 稳定时重复 placement 会替换同一图片，避免流式重绘叠加；
    width/height 为单元格尺寸（s=/v=，对齐 TS imageWidthCells）。
    """
    controls = [f"a=p,f=100,i={int(image_id)}"]
    if width is not None:
        controls.append(f"s={int(width)}")
    if height is not None:
        controls.append(f"v={int(height)}")
    return _encode_chunks(",".join(controls), data, chunk_size)


def encode_kitty_delete(image_id: int) -> str:
    """删除指定 kitty 图片（d=a,i=<id>）。"""
    return f"\x1b_Ga=d,d=i{int(image_id)}\x1b\\"


def encode_iterm2_image(data: bytes, *, name: str = "") -> str:
    """把图片编码为 iTerm2 inline image 序列。"""
    encoded = base64.b64encode(data).decode("ascii")
    return f"\x1b]1337;File=name={name};inline=1:{encoded}\x07"


class TerminalImage(Widget):
    """终端内联图片组件：kitty/iTerm2 序列；不支持时回退占位文本。"""

    def __init__(self, path_or_bytes: str | bytes, name: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._source: str | bytes = path_or_bytes
        self._name: str = name

    def _load(self) -> bytes:
        if isinstance(self._source, bytes):
            return self._source
        with open(self._source, "rb") as handle:
            return handle.read()

    def render_sequence(self) -> str:
        """终端图像传输序列（能力探测后返回）。"""
        try:
            data = self._load()
        except OSError:
            return ""
        capabilities = detect_capabilities()
        if "kitty" in capabilities:
            return encode_kitty_placement(data, image_id=id(self) & 0xFFFFFF)
        if "iterm2" in capabilities:
            return encode_iterm2_image(data, name=self._name)
        return ""

    def render(self, width: int, height: int) -> list[Line]:
        sequence = self.render_sequence()
        if sequence:
            line = blank_line(width, self.base_style)
            line.passthrough = sequence
            return [line]
        source = (
            self._source.decode("utf-8", "replace")
            if isinstance(self._source, bytes)
            else self._source
        )
        label = f"[image: {self._name or source}]"
        return [line_from_text(label, width, self.base_style)]


__all__ = [
    "TerminalImage",
    "detect_capabilities",
    "encode_kitty_image",
    "encode_kitty_placement",
    "encode_kitty_delete",
    "encode_iterm2_image",
]
