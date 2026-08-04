"""剪贴板图片读取与处理（对齐 TS utils/clipboard-image.ts + image-process.ts）。

- 读取：Windows PowerShell / macOS osascript / Linux wl-paste|xclip 多层回退；
- 处理：EXIF 方向校正 + 缩放（max 2000x2000）+ 转 PNG。
"""

from __future__ import annotations

import asyncio
import sys

from pi_agent.tools.image_pipeline import process_image_sync

MAX_IMAGE_DIMENSION = 2000

_WINDOWS_SCRIPT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$img = [System.Windows.Forms.Clipboard]::GetImage();"
    "if ($img) {"
    "$ms = New-Object System.IO.MemoryStream;"
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);"
    "[Console]::OpenStandardOutput().Write($ms.ToArray())}"
)

_MACOS_SCRIPT = (
    "set png_path to (POSIX file (do shell script \"mktemp /tmp/pi-clipboard-XXXX.png\") as text)\n"
    "try\n"
    "  set the clipboard to (read (clipboard info) as «class PNGf»)\n"
    "on error\n"
    "  return\n"
    "end try"
)


class ClipboardImage:
    """跨平台剪贴板图片读取 + Pillow 处理。"""

    @staticmethod
    def build_command() -> list[str]:
        """返回当前平台的剪贴板读取命令（测试可注入）。"""
        if sys.platform == "win32":
            return ["powershell", "-NoProfile", "-Command", _WINDOWS_SCRIPT]
        if sys.platform == "darwin":
            return ["osascript", "-e", _MACOS_SCRIPT]
        return ["wl-paste", "--type", "image/png"]

    @staticmethod
    async def read() -> bytes | None:
        """读取剪贴板图片（PNG bytes）；无图片/失败返回 None。"""
        data = await _run_command(ClipboardImage.build_command())
        if data:
            return data
        if sys.platform.startswith("linux"):
            data = await _run_command(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
            )
            if data:
                return data
        return None

    @staticmethod
    def process(data: bytes) -> bytes:
        """格式转换 + EXIF 校正 + 缩放 → PNG bytes。"""
        result = process_image_sync(
            data, auto_resize=True, max_dimension=MAX_IMAGE_DIMENSION
        )
        if not result["ok"]:
            raise ValueError(result["message"])
        return result["data"]


async def _run_command(args: list[str]) -> bytes | None:
    """执行命令并返回 stdout；失败/超时返回 None。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return None
        if proc.returncode != 0:
            return None
        return stdout if stdout else None
    except (OSError, ValueError):
        return None


__all__ = [
    "ClipboardImage",
    "MAX_IMAGE_DIMENSION",
    "_run_command",
]
