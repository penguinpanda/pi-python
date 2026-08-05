"""展示层渲染工具（对齐 TS `core/tools/render-utils.ts`）。

路径缩短、OSC 8 文件超链接、文本规范化、图片尺寸解析与回退渲染。
Python 侧以纯文本 / OSC 序列输出，不依赖 Textual widget。
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pi_agent.shell_output import sanitize_binary_output
from pi_tui.terminal_image import detect_capabilities

__all__ = [
    "shorten_path",
    "link_path",
    "hyperlink",
    "str_value",
    "replace_tabs",
    "normalize_display_text",
    "strip_ansi",
    "get_text_output",
    "get_image_dimensions",
    "image_fallback",
    "invalid_arg_text",
    "render_tool_path",
]


# 与 TS ansi.ts 相同：OSC 序列（ESC ] ... ST）优先，其次 CSI 及其变体。
_ANSI_PATTERN = re.compile(
    r"(?:\x1b\][\s\S]*?(?:\x07|\x1b\\|\x9c))"
    r"|(?:[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~])"
)


def shorten_path(path: str) -> str:
    """把 home 前缀的绝对路径缩短为 ~/...（非字符串返回空串）。"""
    if not isinstance(path, str):
        return ""
    home = str(Path.home())
    if home and path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def _hyperlink_supported() -> bool:
    """保守探测 OSC 8 超链接支持（对齐 TS detectCapabilities 的子集）。"""
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    terminal_emulator = os.environ.get("TERMINAL_EMULATOR", "").lower()
    term = os.environ.get("TERM", "").lower()

    # tmux / screen 不确定是否转发 OSC 8，保守关闭。
    if os.environ.get("TMUX") or term.startswith("tmux"):
        return False
    if term.startswith("screen"):
        return False
    if os.environ.get("KITTY_WINDOW_ID") or term_program == "kitty":
        return True
    if os.environ.get("GHOSTTY_RESOURCES_DIR") or "ghostty" in term or term_program == "ghostty":
        return True
    if os.environ.get("WEZTERM_PANE") or term_program == "wezterm":
        return True
    if (
        os.environ.get("WARP_SESSION_ID")
        or os.environ.get("WARP_TERMINAL_SESSION_UUID")
        or term_program == "warpterminal"
    ):
        return True
    if os.environ.get("ITERM_SESSION_ID") or term_program == "iterm.app":
        return True
    if os.environ.get("WT_SESSION"):
        return True
    if term_program in ("vscode", "alacritty"):
        return True
    if terminal_emulator == "jetbrains-jediterm":
        return False
    if os.name == "nt":
        return False
    return False


def hyperlink(text: str, url: str) -> str:
    """OSC 8 超链接序列。"""
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def _resolve_path(raw_path: str, cwd: str) -> str:
    """解析为绝对路径（支持 ~ 与 file:// 前缀，对齐 TS resolvePath）。"""
    if raw_path.startswith("file://"):
        return unquote(urlparse(raw_path).path)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    return str(path.resolve())


def link_path(styled_text: str, raw_path: str, cwd: str) -> str:
    """终端支持 OSC 8 时把样式文本包成 file:// 超链接，否则原样返回。"""
    if not _hyperlink_supported():
        return styled_text
    absolute_path = _resolve_path(raw_path, cwd)
    return hyperlink(styled_text, Path(absolute_path).as_uri())


def str_value(value: object) -> str | None:
    """string → string；None → ""；其它 → None（对齐 TS str()）。"""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    """Tab 替换为三个空格（对齐 TS replaceTabs）。"""
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    """去掉 \\r（对齐 TS normalizeDisplayText）。"""
    return text.replace("\r", "")


def strip_ansi(value: str) -> str:
    """去掉 ANSI 转义序列（OSC + CSI，对齐 TS stripAnsi）。"""
    if not isinstance(value, str):
        raise TypeError(f"Expected a `str`, got `{type(value).__name__}`")
    if "\x1b" not in value and "\x9b" not in value:
        return value
    return _ANSI_PATTERN.sub("", value)


def _image_supported() -> bool:
    return bool(detect_capabilities())


def _to_bytes(data: str | bytes) -> bytes:
    if isinstance(data, bytes):
        return data
    try:
        return base64.b64decode(data, validate=False)
    except (ValueError, TypeError):
        return b""


def _png_dimensions(buffer: bytes) -> tuple[int, int] | None:
    if len(buffer) < 24 or buffer[:4] != b"\x89PNG":
        return None
    width = int.from_bytes(buffer[16:20], "big")
    height = int.from_bytes(buffer[20:24], "big")
    return (width, height)


def _jpeg_dimensions(buffer: bytes) -> tuple[int, int] | None:
    if len(buffer) < 2 or buffer[0] != 0xFF or buffer[1] != 0xD8:
        return None
    offset = 2
    while offset < len(buffer) - 9:
        if buffer[offset] != 0xFF:
            offset += 1
            continue
        marker = buffer[offset + 1]
        if 0xC0 <= marker <= 0xC2:
            height = int.from_bytes(buffer[offset + 5 : offset + 7], "big")
            width = int.from_bytes(buffer[offset + 7 : offset + 9], "big")
            return (width, height)
        if offset + 3 >= len(buffer):
            return None
        length = int.from_bytes(buffer[offset + 2 : offset + 4], "big")
        if length < 2:
            return None
        offset += 2 + length
    return None


def _gif_dimensions(buffer: bytes) -> tuple[int, int] | None:
    if len(buffer) < 10:
        return None
    if buffer[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width = int.from_bytes(buffer[6:8], "little")
    height = int.from_bytes(buffer[8:10], "little")
    return (width, height)


def _webp_dimensions(buffer: bytes) -> tuple[int, int] | None:
    if len(buffer) < 30 or buffer[:4] != b"RIFF" or buffer[8:12] != b"WEBP":
        return None
    chunk = buffer[12:16]
    if chunk == b"VP8 ":
        width = int.from_bytes(buffer[26:28], "little") & 0x3FFF
        height = int.from_bytes(buffer[28:30], "little") & 0x3FFF
        return (width, height)
    if chunk == b"VP8L":
        if len(buffer) < 25:
            return None
        bits = int.from_bytes(buffer[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height)
    if chunk == b"VP8X":
        width = int.from_bytes(buffer[24:27], "little") + 1
        height = int.from_bytes(buffer[27:30], "little") + 1
        return (width, height)
    return None


def get_image_dimensions(data: str | bytes, mime_type: str) -> tuple[int, int] | None:
    """解析 PNG / JPEG / GIF / WebP 像素尺寸（对齐 TS getImageDimensions）。"""
    buffer = _to_bytes(data)
    if mime_type == "image/png":
        return _png_dimensions(buffer)
    if mime_type == "image/jpeg":
        return _jpeg_dimensions(buffer)
    if mime_type == "image/gif":
        return _gif_dimensions(buffer)
    if mime_type == "image/webp":
        return _webp_dimensions(buffer)
    return None


def image_fallback(
    mime_type: str,
    dimensions: tuple[int, int] | None = None,
    filename: str | None = None,
) -> str:
    """终端不支持内联图片时的文本回退（对齐 TS imageFallback）。"""
    parts: list[str] = []
    if filename:
        display = shorten_path(filename)
        if _hyperlink_supported() and os.path.isabs(filename):
            parts.append(hyperlink(display, Path(filename).as_uri()))
        else:
            parts.append(display)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions[0]}x{dimensions[1]}")
    return f"[Image: {' '.join(parts)}]"


def get_text_output(result: dict[str, Any] | None, show_images: bool) -> str:
    """把工具结果的 content 块渲染为文本；无图像能力 / 关闭图像时回退指示。"""
    if not result:
        return ""
    content = result.get("content", [])
    text_blocks = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
    image_blocks = [c for c in content if isinstance(c, dict) and c.get("type") == "image"]

    output = "\n".join(
        sanitize_binary_output(strip_ansi(str(c.get("text", "")))).replace("\r", "")
        for c in text_blocks
    )

    if image_blocks and (not _image_supported() or not show_images):
        indicators: list[str] = []
        for image in image_blocks:
            mime_type = str(image.get("mimeType") or "image/unknown")
            dimensions = None
            data = image.get("data")
            if data:
                dimensions = get_image_dimensions(data, mime_type)
            indicators.append(image_fallback(mime_type, dimensions))
        joined = "\n".join(indicators)
        output = f"{output}\n{joined}" if output else joined
    return output


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    hex_value = value.lstrip("#")
    if len(hex_value) != 6:
        return (0, 0, 0)
    try:
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
    except ValueError:
        return (0, 0, 0)
    return (r, g, b)


def _theme_fg(theme: object, name: str, text: str) -> str:
    """按主题色名生成前景色：优先 theme.fg(name, text)（ThemeFacade），
    退回到 theme.colors 字典 + ANSI 24bit 色（pi_tui.Theme）。"""
    fg = getattr(theme, "fg", None)
    if callable(fg):
        return fg(name, text)
    colors = getattr(theme, "colors", None)
    if isinstance(colors, dict):
        color = colors.get(name)
        if isinstance(color, str) and color:
            r, g, b = _hex_to_rgb(color)
            return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"
    return text


def invalid_arg_text(theme: object) -> str:
    """无效参数的红色占位文本（对齐 TS invalidArgText）。"""
    return _theme_fg(theme, "error", "[invalid arg]")


def render_tool_path(
    raw_path: str | None,
    theme: object,
    cwd: str,
    options: dict[str, Any] | None = None,
) -> str:
    """渲染工具路径：缩短 + accent 着色 + OSC 8 文件链接（对齐 TS renderToolPath）。"""
    if raw_path is None:
        return invalid_arg_text(theme)
    value = raw_path or (options or {}).get("emptyFallback")
    if not value:
        return _theme_fg(theme, "toolOutput", "...")
    return link_path(_theme_fg(theme, "accent", shorten_path(value)), value, cwd)
