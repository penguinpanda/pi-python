"""OSC 8 超链接工具（纯 pi_tui，无应用层依赖）。"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .engine.cells import Line

_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_~/])((?:[A-Za-z]:[\\/]|/)[^\s()\[\]{}<>\"']*?)"
    r"(?=[\s,;:)\]}>\"']|$)"
)


def osc8_hyperlink_supported() -> bool:
    """保守探测 OSC 8 超链接支持（对齐 TS detectCapabilities 子集）。"""
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    terminal_emulator = os.environ.get("TERMINAL_EMULATOR", "").lower()
    term = os.environ.get("TERM", "").lower()
    # tmux / screen 不确定是否转发 OSC 8，保守关闭。
    if os.environ.get("TMUX") or term.startswith("tmux") or term.startswith("screen"):
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
    return False


def linkify_paths(text: str) -> str:
    """把存在的绝对路径包装为 Rich `[link=file://...]path[/link]` 标记。

    仅当终端支持 OSC 8 且路径在磁盘上存在时才包装，避免误伤普通文本。
    """
    if not text or not osc8_hyperlink_supported():
        return text
    parts: list[str] = []
    last = 0
    for match in _PATH_PATTERN.finditer(text):
        raw = match.group(1)
        if not raw:
            continue
        try:
            path = Path(raw).expanduser()
        except (OSError, ValueError):
            continue
        if not path.exists():
            continue
        parts.append(text[last : match.start()])
        parts.append(f"[link={path.resolve().as_uri()}]{raw}[/link]")
        last = match.end()
    if not parts:
        return text
    parts.append(text[last:])
    return "".join(parts)


def linkify_lines(lines: list[Line]) -> None:
    """在已渲染的 Line 上给存在的绝对路径单元格设置 OSC8 链接（保留 markdown）。"""
    if not osc8_hyperlink_supported():
        return
    for line in lines:
        text = line.text()
        for match in _PATH_PATTERN.finditer(text):
            raw = match.group(1)
            if not raw:
                continue
            try:
                path = Path(raw).expanduser()
            except (OSError, ValueError):
                continue
            if not path.exists():
                continue
            uri = path.resolve().as_uri()
            start, end = match.start(1), match.end(1)
            for index in range(start, min(end, len(line.cells))):
                line.cells[index].link = uri


def has_abs_paths(text: str) -> bool:
    return _PATH_PATTERN.search(text) is not None


def normalize_path_slashes(text: str) -> str:
    """把绝对路径里的反斜杠归一为斜杠（防止 markdown 转义吞掉路径）。"""
    if "\\" not in text:
        return text
    return _PATH_PATTERN.sub(lambda match: match.group(1).replace("\\", "/"), text)


__all__ = [
    "osc8_hyperlink_supported",
    "linkify_paths",
    "linkify_lines",
    "normalize_path_slashes",
    "has_abs_paths",
]
