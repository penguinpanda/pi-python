"""终端能力：OSC 11 背景色查询（尽力而为，跨平台）。"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import cast

_OSC11_PATTERN = re.compile(r"\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)", re.IGNORECASE)

# 是否发过 OSC 11 查询且尚未收到响应（退出 TUI 前用于兜底 drain）。
_OSC11_QUERY_PENDING = False


def _parse_osc_hex_channel(channel: str) -> int | None:
    """把 1/2/4 位十六进制通道归一化到 0-255（对齐 TS parseOscHexChannel）。"""
    if not re.fullmatch(r"[0-9a-f]+", channel, re.IGNORECASE):
        return None
    max_value = 16 ** len(channel) - 1
    if max_value <= 0:
        return None
    return round((int(channel, 16) / max_value) * 255)


def parse_osc11_background(data: str) -> tuple[int, int, int] | None:
    """解析 OSC 11 背景色响应：支持 #RRGGBB / #RRRRGGGGBBBB / rgb:r/g/b。"""
    match = _OSC11_PATTERN.search(data)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if re.fullmatch(r"[0-9a-f]{6}", hex_value, re.IGNORECASE):
            channels = [int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]
            return (channels[0], channels[1], channels[2])
        if re.fullmatch(r"[0-9a-f]{12}", hex_value, re.IGNORECASE):
            parsed = [_parse_osc_hex_channel(hex_value[i : i + 4]) for i in (0, 4, 8)]
            if all(channel is not None for channel in parsed):
                return cast(tuple[int, int, int], (parsed[0], parsed[1], parsed[2]))
            return None
        return None
    rgb_value = re.sub(r"^rgba?:", "", value, flags=re.IGNORECASE)
    parts = rgb_value.split("/")
    if len(parts) != 3:
        return None
    parsed = [_parse_osc_hex_channel(part) for part in parts]
    if all(channel is not None for channel in parsed):
        return cast(tuple[int, int, int], (parsed[0], parsed[1], parsed[2]))
    return None


def _read_osc_response_posix(fd: int, timeout: float, drain: float) -> str:
    """POSIX：在 raw 模式下读取终端响应直到 BEL 或 ST（调用方负责 setraw）。"""
    buffer = bytearray()
    deadline = time.monotonic() + timeout + drain
    import select

    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 256)
        if not chunk:
            break
        buffer.extend(chunk)
        if b"\x07" in chunk or b"\x1b\\" in chunk:
            break
    return buffer.decode("latin-1", "replace")


def _read_osc_response_windows(timeout: float, drain: float) -> str:
    """Windows：msvcrt 读取控制台输入直到 BEL 或 ST。"""
    import msvcrt

    buffer = bytearray()
    deadline = time.monotonic() + timeout + drain
    while time.monotonic() < deadline:
        if not msvcrt.kbhit():
            continue
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            # 功能键前缀：跳过后续扫描码。
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue
        code = ord(char)
        if code <= 0xFF:
            buffer.append(code)
        else:
            buffer.extend(char.encode("latin-1", "replace"))
        if b"\x07" in buffer or buffer.endswith(b"\x1b\\"):
            break
    return buffer.decode("latin-1", "replace")


def query_terminal_background(timeout: float = 0.5) -> tuple[int, int, int] | None:
    """启动时查询终端背景色（OSC 11）；非 TTY / 失败返回 None。

    POSIX 下先进入 raw 模式（关闭回显）再写查询：若终端支持 OSC 11，
    它注入的响应会因回显开启而被原样显示到主屏——这正是退出 pi 后看到
    OSC 11 响应文本的原因。raw 模式下响应只进输入队列被我们消费。
    """
    global _OSC11_QUERY_PENDING
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return None
        if os.environ.get("TERM", "").lower() == "dumb":
            return None
        # 直接写 fd，绕开 Python stdout 缓冲，避免查询序列被延迟刷出。
        _OSC11_QUERY_PENDING = True
        if os.name == "posix":
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)  # type: ignore[attr-defined]
            try:
                tty.setraw(fd)  # type: ignore[attr-defined]
                os.write(sys.stdout.fileno(), b"\x1b]11;?\x07")
                response = _read_osc_response_posix(fd, timeout, 0.25)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)  # type: ignore[attr-defined]
        else:
            os.write(sys.stdout.fileno(), b"\x1b]11;?\x07")
            response = _read_osc_response_windows(timeout, 0.25)
        parsed = parse_osc11_background(response)
        if parsed is not None:
            _OSC11_QUERY_PENDING = False
        return parsed
    except Exception:
        return None


def drain_pending_osc_response(timeout: float = 0.2) -> None:
    """退出 TUI 前消费可能晚到的 OSC 11 响应，避免残留输入漏到 shell。"""
    global _OSC11_QUERY_PENDING
    if not _OSC11_QUERY_PENDING:
        return
    try:
        if not sys.stdin.isatty():
            return
        if os.name == "posix":
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)  # type: ignore[attr-defined]
            try:
                tty.setraw(fd)  # type: ignore[attr-defined]
                _read_osc_response_posix(fd, timeout, 0.0)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)  # type: ignore[attr-defined]
        else:
            _read_osc_response_windows(timeout, 0.0)
    except Exception:
        pass
    finally:
        _OSC11_QUERY_PENDING = False
