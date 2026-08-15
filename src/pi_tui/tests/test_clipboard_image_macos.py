"""macOS 剪贴板脚本回归测试。"""

from __future__ import annotations

import os

import pytest

from pi_tui import clipboard_image as ci


def test_macos_script_writes_png_to_file() -> None:
    """脚本必须把 PNG 写入文件；旧的空操作写法(重写回剪贴板)已移除。"""
    assert "the clipboard as" in ci._MACOS_SCRIPT
    assert "open for access" in ci._MACOS_SCRIPT
    assert "write png_data to f" in ci._MACOS_SCRIPT
    assert "set the clipboard to" not in ci._MACOS_SCRIPT


@pytest.mark.asyncio
async def test_read_macos_reads_temp_file_and_cleans(monkeypatch) -> None:
    target = os.path.join(os.getcwd(), "pi-test-clip-tmp.png")
    monkeypatch.setattr(ci, "_MACOS_TMP_PATH", target)

    async def fake_run(args):
        with open(target, "wb") as f:
            f.write(b"PNGDATA")
        return None

    monkeypatch.setattr(ci, "_run_command", fake_run)
    data = await ci._read_macos()
    assert data == b"PNGDATA"
    assert not os.path.exists(target)
