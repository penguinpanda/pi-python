"""图片管线测试（EXIF / 缩放 / 多格式转换）。"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from pi_agent.tools.image_pipeline import (
    convert_image,
    exif_orientation,
    process_image_sync,
    resize_image,
)


def _png_bytes(size=(32, 24), color=(255, 0, 0)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_with_orientation(orientation: int, size=(200, 100)) -> bytes:
    image = Image.new("RGB", size, (0, 128, 255))
    exif = Image.Exif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def _bmp_bytes(size=(40, 30), color=(0, 255, 0)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    return buffer.getvalue()


class TestImagePipeline:
    def test_process_jpeg_converts_to_png(self):
        result = process_image_sync(_jpeg_with_orientation(1), "image/jpeg")
        assert result["ok"] is True
        assert result["mimeType"] == "image/png"
        with Image.open(io.BytesIO(result["data"])) as image:
            assert image.format == "PNG"
            assert image.size == (200, 100)

    def test_exif_orientation_applied(self):
        data = _jpeg_with_orientation(6)
        result = process_image_sync(data, "image/jpeg")
        assert result["ok"] is True
        with Image.open(io.BytesIO(result["data"])) as image:
            # orientation=6 旋转 90°，宽高互换。
            assert image.size == (100, 200)

    def test_exif_orientation_helper_preserves_format(self):
        data = _jpeg_with_orientation(6)
        oriented = exif_orientation(data)
        with Image.open(io.BytesIO(oriented)) as image:
            assert image.format == "JPEG"
            assert image.size == (100, 200)

    def test_resize_large_image(self):
        data = _png_bytes(size=(4000, 2000))
        resized = resize_image(data, max_dimension=2000)
        with Image.open(io.BytesIO(resized)) as image:
            assert max(image.size) == 2000
            assert image.size == (2000, 1000)

    def test_resize_small_image_unchanged(self):
        data = _png_bytes(size=(100, 100))
        assert resize_image(data) == data

    def test_convert_bmp_to_png(self):
        data = _bmp_bytes()
        converted, mime_type = convert_image(data, "PNG")
        assert mime_type == "image/png"
        with Image.open(io.BytesIO(converted)) as image:
            assert image.format == "PNG"
            assert image.size == (40, 30)

    def test_invalid_data_returns_error_dict(self):
        result = process_image_sync(b"not an image", "image/png")
        assert result["ok"] is False
        assert "failed" in result["message"].lower()

    def test_auto_resize_hint(self):
        data = _png_bytes(size=(3000, 1500))
        result = process_image_sync(data, "image/png", auto_resize=True)
        assert result["ok"] is True
        assert result["hints"]
        with Image.open(io.BytesIO(result["data"])) as image:
            assert max(image.size) == 2000


class TestReadToolPipeline:
    @pytest.mark.asyncio
    async def test_read_tool_converts_bmp_by_default(self, tmp_path):
        from pi_agent import PythonExecutionEnv, create_read_tool

        (tmp_path / "pic.bmp").write_bytes(_bmp_bytes())
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()

        class _Context:
            pass

        context = _Context()
        context.env = env
        result = await tool.execute("t1", {"path": "pic.bmp"}, None, None, context)
        texts = [block.get("text", "") for block in result.content]
        assert any("Read image file [image/png]" in text for text in texts)
        assert not any("Image omitted" in text for text in texts)
        assert any(block.get("type") == "image" for block in result.content)
