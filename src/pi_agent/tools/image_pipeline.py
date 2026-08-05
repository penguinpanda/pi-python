"""图片处理管线（对齐 TS utils/photon + image-resize + image-convert + exif-orientation）。

统一提供：EXIF 方向校正、缩放（max 2000x2000）、多格式转 PNG。
read 工具默认接入；剪贴板图片复用同一实现。
"""

from __future__ import annotations

import asyncio
import io

from PIL import Image, ImageOps

from .image import detect_supported_image_mime_type

MAX_IMAGE_DIMENSION = 2000


def exif_orientation(data: bytes) -> bytes:
    """校正 EXIF 方向并返回同格式字节（无法解析时原样返回）。"""
    try:
        with Image.open(io.BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            if oriented is image:
                return data
            buffer = io.BytesIO()
            oriented.save(buffer, format=image.format or "PNG")
            return buffer.getvalue()
    except Exception:
        return data


def resize_image(data: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> bytes:
    """等比缩放到 max_dimension 内，返回原格式字节。"""
    try:
        with Image.open(io.BytesIO(data)) as image:
            if max(image.size) <= max_dimension:
                return data
            resized = image.copy()
            resized.thumbnail((max_dimension, max_dimension))
            buffer = io.BytesIO()
            resized.save(buffer, format=image.format or "PNG")
            return buffer.getvalue()
    except Exception:
        return data


def convert_image(data: bytes, target_format: str = "PNG") -> tuple[bytes, str]:
    """转换为目标格式；返回 (bytes, mime_type)。"""
    with Image.open(io.BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened)
        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        output = image.convert("RGBA" if has_alpha else "RGB")
        buffer = io.BytesIO()
        output.save(buffer, format=target_format)
        mime_type = (
            "image/png" if target_format.upper() == "PNG" else f"image/{target_format.lower()}"
        )
        return buffer.getvalue(), mime_type


def process_image_sync(
    data: bytes,
    mime_type: str | None = None,
    *,
    auto_resize: bool = True,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> dict:
    """同步处理：EXIF 校正 + 缩放 + 转 PNG。

    返回 {"ok", "data", "mimeType", "hints", "message"}（read 工具 image_processor 契约）。
    """
    try:
        if mime_type is None:
            mime_type = detect_supported_image_mime_type(data) or "image/png"
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            hints: list[str] = []
            if auto_resize and max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension))
                hints.append(f"Image resized to fit within {max_dimension}x{max_dimension}")
            has_alpha = image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            )
            output = image.convert("RGBA" if has_alpha else "RGB")
            buffer = io.BytesIO()
            output.save(buffer, format="PNG")
        return {
            "ok": True,
            "data": buffer.getvalue(),
            "mimeType": "image/png",
            "hints": hints,
            "message": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "mimeType": mime_type or "image/png",
            "hints": [],
            "message": f"Image processing failed: {exc}",
        }


async def process_image(
    data: bytes,
    mime_type: str | None = None,
    options: dict | None = None,
) -> dict:
    """异步处理入口（read 工具 image_processor 契约）。"""
    options = options or {}
    return await asyncio.to_thread(
        process_image_sync,
        data,
        mime_type,
        auto_resize=bool(options.get("autoResizeImages", True)),
    )


__all__ = [
    "MAX_IMAGE_DIMENSION",
    "exif_orientation",
    "resize_image",
    "convert_image",
    "process_image_sync",
    "process_image",
]
