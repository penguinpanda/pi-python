"""终端图像协议（kitty / iTerm2）序列生成，独立可单测。"""

from __future__ import annotations

import base64
import os
import random
import re

from pi_tui.engine.cells import Line, blank_line, line_from_text
from pi_tui.engine.widgets import Widget

KITTY_PREFIX = "\x1b_G"
ITERM2_PREFIX = "\x1b]1337;File="

_kitty_metadata: dict[int, dict[str, int]] = {}


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


def is_image_line(line: str) -> bool:
    """检测一行是否为 kitty/iTerm2 图片序列（含多行图片的游标前缀）。"""
    return line.startswith((KITTY_PREFIX, ITERM2_PREFIX)) or (
        KITTY_PREFIX in line or ITERM2_PREFIX in line
    )


def allocate_image_id() -> int:
    """生成随机 image id, 避免不同模块实例之间的冲突。"""
    return random.randint(1, 0xFFFFFFFF)


def register_kitty_image_metadata(metadata: dict[str, int]) -> None:
    """登记 kitty 图片元数据, 供行级裁剪使用; 上限 1000 条。"""
    image_id = int(metadata["imageId"])
    _kitty_metadata[image_id] = dict(metadata)
    while len(_kitty_metadata) > 1000:
        _kitty_metadata.pop(next(iter(_kitty_metadata)))


def get_kitty_image_metadata(line: str) -> dict[str, int] | None:
    """从行内 kitty 控制序列读取已登记的图片元数据。"""
    match = re.search(r"\x1b_G([^;]*);", line)
    if match is None:
        return None
    image_match = re.search(r"(?:^|,)i=(\d+)(?:,|$)", match.group(1))
    if image_match is None:
        return None
    return _kitty_metadata.get(int(image_match.group(1)))


def crop_kitty_image_line(line: str, hidden_rows: int, visible_rows: int) -> str:
    """裁剪 kitty placement 行的可见行区域 (对齐 TS cropKittyImageLine)。"""
    metadata = get_kitty_image_metadata(line)
    match = re.search(r"\x1b_G([^;]*);", line)
    if metadata is None or match is None:
        return line
    rows = int(metadata["rows"])
    if hidden_rows < 0 or hidden_rows >= rows or visible_rows <= 0:
        return line
    cropped_rows = min(visible_rows, rows - hidden_rows)
    if hidden_rows == 0 and cropped_rows == rows:
        return line
    height_px = int(metadata["heightPx"])
    source_y = (height_px * hidden_rows) // rows
    source_end = (height_px * (hidden_rows + cropped_rows) + rows - 1) // rows
    source_height = max(1, min(height_px, source_end) - source_y)
    controls = [part for part in match.group(1).split(",") if not re.fullmatch(r"[yhr]=.*", part)]
    controls.append(f"y={source_y}")
    controls.append(f"h={source_height}")
    controls.append(f"r={cropped_rows}")
    return f"{line[: match.start()]}{KITTY_PREFIX}{','.join(controls)};{line[match.end() :]}"


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
    "KITTY_PREFIX",
    "ITERM2_PREFIX",
    "TerminalImage",
    "detect_capabilities",
    "is_image_line",
    "allocate_image_id",
    "register_kitty_image_metadata",
    "get_kitty_image_metadata",
    "crop_kitty_image_line",
    "encode_kitty_image",
    "encode_kitty_placement",
    "encode_kitty_delete",
    "encode_iterm2_image",
]
