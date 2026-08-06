"""终端 I/O：raw 模式 / alt-screen / 尺寸 / 差分写入 / OSC。

POSIX 与 Windows 双平台；FakeTerminal 提供无 TTY 的测试驱动。
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from typing import Any

from .cells import Cell, Line, _visible_slice, line_to_ansi
from .keys import KeyEvent


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no")


class ScreenBuffer:
    """屏幕帧缓冲：按行差分生成写入命令。"""

    def __init__(self, width: int = 80, height: int = 24) -> None:
        self.width = width
        self.height = height
        self._lines: list[Line] = []
        self._dirty = True

    def resize(self, width: int, height: int) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self._dirty = True

    def _normalize(self, lines: list[Line]) -> list[Line]:
        normalized: list[Line] = []
        for line in lines:
            cells, used = _visible_slice(line.cells, self.width)
            if used < self.width:
                cells.extend(Cell(" ") for _ in range(self.width - used))
            normalized.append(Line(cells, passthrough=line.passthrough))
        while len(normalized) < self.height:
            normalized.append(Line([Cell(" ") for _ in range(self.width)]))
        return normalized[: self.height]

    def diff(self, lines: list[Line]) -> str:
        """根据新帧输出 ANSI 命令；仅重画变化的行。"""
        normalized = self._normalize(lines)
        if self._dirty:
            self._lines = normalized
            self._dirty = False
            return "\x1b[2J\x1b[H" + "\r\n".join(
                f"{line.passthrough}{line_to_ansi(line, self.width)}" for line in normalized
            )
        parts_out: list[str] = []
        previous = self._lines
        for index in range(self.height):
            if previous[index] != normalized[index]:
                line = normalized[index]
                parts_out.append(
                    f"\x1b[{index + 1};1H{line.passthrough}{line_to_ansi(line, self.width)}"
                )
        self._lines = normalized
        return "".join(parts_out)

    def reset(self) -> None:
        self._lines = []
        self._dirty = True


class Terminal:
    """真实终端驱动：raw 模式、alt-screen、尺寸查询、输入读取。"""

    def __init__(
        self,
        *,
        stdin: Any = None,
        stdout: Any = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._size = size
        self._old_termios: Any = None
        self._old_console_mode: int | None = None
        self._entered = False
        self.alt_screen = True
        self.raw_buffer: bytearray = bytearray()
        self.sync_output = _env_flag("PI_SYNC_OUTPUT", default=True)

    @property
    def size(self) -> tuple[int, int]:
        if self._size is not None:
            return self._size
        return self.query_size()

    def query_size(self) -> tuple[int, int]:
        """查询终端尺寸（POSIX ioctl；Windows console API；默认 80x24）。"""
        try:
            if os.name == "posix":
                import fcntl
                import struct
                import termios

                data = fcntl.ioctl(  # type: ignore[attr-defined]
                    self.stdin.fileno(),
                    termios.TIOCGWINSZ,  # type: ignore[attr-defined]
                    b"\0" * 8,
                )
                rows, cols = struct.unpack("HHHH", data)[:2]
                if rows and cols:
                    return (cols, rows)
            elif os.name == "nt":
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                info = ctypes.create_string_buffer(22)
                if kernel32.GetConsoleScreenBufferInfo(handle, info):
                    raw = bytes(info)
                    # CONSOLE_SCREEN_BUFFER_INFO: dwSize(0-3) dwCursorPosition(4-7)
                    # wAttributes(8-9) srWindow(10-17) dwMaximumWindowSize(18-21)。
                    # 可见尺寸取 srWindow 的 Right/Bottom（offset 14/16）。
                    right = int.from_bytes(raw[14:16], "little")
                    bottom = int.from_bytes(raw[16:18], "little")
                    return (right + 1, bottom + 1)
        except Exception:
            pass
        return (80, 24)

    async def enter(self, *, alt_screen: bool = True) -> None:
        """进入 TUI 模式：raw 输入 + （可选）alt-screen + 隐藏光标 + 鼠标/粘贴。

        regular 模式（对齐 TS TuiMainScreen）不进 alt-screen、不启用鼠标。
        """
        if self._entered:
            return
        self._entered = True
        self.alt_screen = alt_screen
        if os.name == "posix":
            import termios
            import tty

            fd = self.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)  # type: ignore[attr-defined]
            tty.setraw(fd)  # type: ignore[attr-defined]
        elif os.name == "nt":
            self._enable_windows_vt()
        if alt_screen:
            self.write(
                "\x1b[?1049h"  # alt-screen
                "\x1b[?25l"  # 隐藏光标（软件光标由引擎绘制）
                "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h"  # 鼠标 press/drag/hover + SGR
                "\x1b[?1004h"  # 焦点事件（对齐 TS）
                "\x1b[?2004h"  # bracketed paste
                "\x1b]133;A\x07"  # OSC 133 prompt 开始
                "\x1b[>7u\x1b[?u\x1b[c"  # kitty 键盘协议协商（对齐 TS）
            )
        else:
            self.write(
                "\x1b[?25l"
                "\x1b[?2004h"  # bracketed paste
                "\x1b[>7u\x1b[?u\x1b[c"  # kitty 键盘协议协商（对齐 TS）
            )
        if self.sync_output:
            self.write("\x1b[?2026h")  # 同步输出
        self.flush()

    async def exit(self, *, alt_screen: bool | None = None) -> None:
        if not self._entered:
            return
        self._entered = False
        if alt_screen is None:
            alt_screen = self.alt_screen
        if alt_screen:
            self.write(
                "\x1b]133;B\x07"
                "\x1b[?2026l"
                "\x1b[?2004l"
                "\x1b[?1006l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
                "\x1b[?25h"
                "\x1b[?1049l"
            )
        else:
            self.write("\x1b]133;B\x07\x1b[?2026l\x1b[?2004l\x1b[?25h")
        self.flush()
        if os.name == "posix":
            import termios

            if self._old_termios is not None:
                try:
                    termios.tcsetattr(  # type: ignore[attr-defined]
                        self.stdin.fileno(),
                        termios.TCSADRAIN,  # type: ignore[attr-defined]
                        self._old_termios,
                    )
                except Exception:
                    pass
        elif os.name == "nt":
            self._restore_windows_console()

    def write(self, data: str) -> None:
        self.stdout.write(data)

    def flush(self) -> None:
        try:
            self.stdout.flush()
        except Exception:
            pass

    def copy_to_clipboard(self, text: str) -> bool:
        """OSC 52 剪贴板写入；返回是否写入。"""
        try:
            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self.write(f"\x1b]52;c;{encoded}\x07")
            self.flush()
            return True
        except Exception:
            return False

    def set_hardware_cursor(self, row: int, col: int) -> None:
        """硬件光标定位（1-based）。"""
        self.write(f"\x1b[{row};{col}H")

    def show_cursor(self) -> None:
        """显示硬件光标（IME 候选窗口需要光标位置锚定）。"""
        self.write("\x1b[?25h")

    def hide_cursor(self) -> None:
        """隐藏硬件光标（软件光标由引擎绘制）。"""
        self.write("\x1b[?25l")

    def set_color_scheme_notifications(self, enabled: bool) -> None:
        """启用/关闭颜色方案通知（?2031h，对齐 TS setTerminalColorSchemeNotifications）。"""
        if not self._entered:
            return
        self.write("\x1b[?2031h" if enabled else "\x1b[?2031l")
        self.flush()

    def set_progress(self, active: bool) -> None:
        """OSC 9;4 终端任务栏进度（对齐 TS setProgress）。"""
        if not self._entered:
            return
        self.write("\x1b]9;4;3\x07" if active else "\x1b]9;4;0\x07")
        self.flush()

    def resize_event(self) -> KeyEvent | None:
        """尺寸变化 → resize 事件（POSIX 信号由 App 注册，Windows 轮询）。"""
        size = self.query_size()
        if size != self._size:
            old = self._size
            self._size = size
            if old is not None:
                return KeyEvent(type="resize", width=size[0], height=size[1])
        return None

    # ------------------------------------------------------------------
    # Windows console mode
    # ------------------------------------------------------------------

    def _enable_windows_vt(self) -> None:
        import ctypes
        import msvcrt

        msvcrt.setmode(self.stdin.fileno(), os.O_BINARY)  # type: ignore[attr-defined]
        msvcrt.setmode(self.stdout.fileno(), os.O_BINARY)  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        in_handle = kernel32.GetStdHandle(-10)
        out_handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(in_handle, ctypes.byref(mode)):
            self._old_console_mode = mode.value
            # 保留 VT input + processed input，关闭行缓冲与回显（对齐 raw 模式）。
            new_mode = mode.value | 0x0200 | 0x0001
            new_mode &= ~(0x0002 | 0x0004)
            kernel32.SetConsoleMode(in_handle, new_mode)
        if kernel32.GetConsoleMode(out_handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(
                out_handle,
                mode.value | 0x0004 | 0x0001 | 0x0002,  # VT processing
            )

    def _restore_windows_console(self) -> None:
        import ctypes

        if self._old_console_mode is None:
            return
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        in_handle = kernel32.GetStdHandle(-10)
        kernel32.SetConsoleMode(in_handle, self._old_console_mode)

    # ------------------------------------------------------------------
    # 输入读取
    # ------------------------------------------------------------------

    async def read_chunk(self) -> bytes | None:
        """读取一块输入（None 表示 EOF）。"""
        if os.name == "posix":
            return await asyncio.to_thread(self._read_posix_chunk)
        return await asyncio.to_thread(self._read_windows_chunk)

    def _read_posix_chunk(self) -> bytes | None:
        """POSIX 读取：select 超时避免线程永久阻塞。

        退出时 asyncio.run 会等待默认线程池线程结束；若 os.read 永久阻塞，
        进程会一直不退出（直到用户按键），shell 提示符也不出现。
        """
        import select

        try:
            ready, _w, _x = select.select([self.stdin.fileno()], [], [], 0.2)
        except (OSError, ValueError):
            return None
        if not ready:
            return b""
        try:
            return os.read(self.stdin.fileno(), 4096)
        except OSError:
            return None

    def _read_windows_chunk(self) -> bytes | None:
        import msvcrt
        import time

        if not msvcrt.kbhit():  # type: ignore[attr-defined]
            time.sleep(0.02)
            return b""
        return os.read(self.stdin.fileno(), 4096)


class FakeTerminal:
    """无 TTY 测试驱动：注入输入字节、捕获输出。"""

    def __init__(self, size: tuple[int, int] = (80, 24)) -> None:
        self._size = size
        self.output: list[str] = []
        self.clipboard: list[str] = []
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._entered = False
        self.sync_output = False
        self.progress = False
        self.color_scheme_notifications = False

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def set_size(self, size: tuple[int, int]) -> None:
        self._size = size

    async def enter(self, *, alt_screen: bool = True) -> None:
        self._entered = True
        self.alt_screen = alt_screen

    async def exit(self, *, alt_screen: bool | None = None) -> None:
        self._entered = False

    def write(self, data: str) -> None:
        self.output.append(data)

    def flush(self) -> None:
        pass

    def copy_to_clipboard(self, text: str) -> bool:
        self.clipboard.append(text)
        return True

    def set_hardware_cursor(self, row: int, col: int) -> None:
        self.write(f"\x1b[{row};{col}H")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def set_color_scheme_notifications(self, enabled: bool) -> None:
        self.color_scheme_notifications = bool(enabled)

    def set_progress(self, active: bool) -> None:
        self.progress = bool(active)

    def resize_event(self) -> KeyEvent | None:
        return None

    def feed(self, data: bytes) -> None:
        self._queue.put_nowait(bytes(data))

    def feed_text(self, text: str) -> None:
        self.feed(text.encode("utf-8"))

    def close(self) -> None:
        if not self._queue.full():
            self._queue.put_nowait(None)

    async def read_chunk(self) -> bytes | None:
        return await self._queue.get()

    @property
    def output_text(self) -> str:
        return "".join(self.output)

    def reset_output(self) -> None:
        self.output = []


__all__ = ["Terminal", "FakeTerminal", "ScreenBuffer", "_env_flag"]
