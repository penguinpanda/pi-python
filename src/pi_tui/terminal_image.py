"""终端图像协议（kitty / iTerm2）序列生成，独立可单测。"""

from __future__ import annotations

import base64
import os

from textual.widgets import Static


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


def encode_kitty_image(
    data: bytes,
    *,
    width: int | None = None,
    height: int | None = None,
    chunk_size: int = 4096,
) -> str:
    """把图片编码为 kitty graphics protocol 序列（单图传输，无 placement）。"""
    encoded = base64.b64encode(data).decode("ascii")
    controls = ["a=T", "f=100"]
    if width is not None:
        controls.append(f"s={int(width)}")
    if height is not None:
        controls.append(f"v={int(height)}")
    header = ",".join(controls)
    parts: list[str] = []
    for index in range(0, len(encoded), chunk_size):
        chunk = encoded[index : index + chunk_size]
        more = "0" if index + chunk_size >= len(encoded) else "1"
        parts.append(f"\x1b_G{header},m={more};{chunk}\x1b\\")
    return "".join(parts)


def encode_iterm2_image(data: bytes, *, name: str = "") -> str:
    """把图片编码为 iTerm2 inline image 序列。"""
    encoded = base64.b64encode(data).decode("ascii")
    return f"\x1b]1337;File=name={name};inline=1:{encoded}\x07"


class TerminalImage(Static):
    """终端内联图片组件：kitty/iTerm2 序列；不支持时回退占位文本。"""

    def __init__(self, path_or_bytes: str | bytes, name: str = "") -> None:
        super().__init__()
        self._source: str | bytes = path_or_bytes
        self._name: str = name

    def _load(self) -> bytes:
        if isinstance(self._source, bytes):
            return self._source
        with open(self._source, "rb") as handle:
            return handle.read()

    def render(self) -> str:
        source = (
            self._source.decode("utf-8", "replace")
            if isinstance(self._source, bytes)
            else self._source
        )
        fallback = f"[image: {self._name or source}]"
        try:
            data = self._load()
        except OSError:
            return fallback
        capabilities = detect_capabilities()
        if "kitty" in capabilities:
            return encode_kitty_image(data)
        if "iterm2" in capabilities:
            return encode_iterm2_image(data, name=self._name)
        return fallback
