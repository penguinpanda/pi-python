"""剪贴板图片读取与处理测试。"""

from __future__ import annotations

import io

from PIL import Image

from pi_agent.tools.image_pipeline import process_image_sync
from pi_tui.clipboard_image import (
    MAX_IMAGE_DIMENSION,
    ClipboardImage,
    _run_command,
)


def _make_png(width=100, height=80, mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestProcess:
    def test_roundtrip(self):
        processed = ClipboardImage.process(_make_png(), process_image_sync)
        with Image.open(io.BytesIO(processed)) as image:
            assert image.format == "PNG"
            assert image.size == (100, 80)

    def test_resizes_large(self):
        data = _make_png(4000, 2000)
        processed = ClipboardImage.process(data, process_image_sync)
        with Image.open(io.BytesIO(processed)) as image:
            width, height = image.size
            assert max(width, height) <= MAX_IMAGE_DIMENSION
            assert height <= MAX_IMAGE_DIMENSION

    def test_preserves_alpha(self):
        processed = ClipboardImage.process(_make_png(mode="RGBA"), process_image_sync)
        with Image.open(io.BytesIO(processed)) as image:
            assert image.mode == "RGBA"

    def test_invalid_data_raises(self):
        import pytest

        with pytest.raises(Exception):  # noqa: B017 - 验证非法输入会抛异常即可
            ClipboardImage.process(b"not an image", process_image_sync)


class TestRead:
    async def test_read_returns_png(self, monkeypatch):
        expected = _make_png()

        async def fake_run(args):
            return expected

        monkeypatch.setattr("pi_tui.clipboard_image._run_command", fake_run)
        assert await ClipboardImage.read() == expected

    async def test_read_returns_none_on_failure(self, monkeypatch):
        async def fake_run(args):
            return None

        monkeypatch.setattr("pi_tui.clipboard_image._run_command", fake_run)
        assert await ClipboardImage.read() is None

    async def test_run_command_success(self):
        import sys

        result = await _run_command([sys.executable, "-c", "print('png-bytes')"])
        assert result is not None
        assert b"png-bytes" in result

    async def test_run_command_missing(self):
        assert await _run_command(["definitely-not-a-real-command-xyz"]) is None
