"""图片处理管线（对齐 TS utils/photon + image-resize + image-convert + exif-orientation）。

统一提供：EXIF 方向校正、缩放（max 2000x2000）、多格式转 PNG。
read 工具默认接入；剪贴板图片复用同一实现。
"""

from __future__ import annotations

import asyncio
import base64
import io
import math

from PIL import Image, ImageOps

from .image import detect_supported_image_mime_type

MAX_IMAGE_DIMENSION = 2000
DEFAULT_MAX_BASE64_BYTES = int(4.5 * 1024 * 1024)
DEFAULT_JPEG_QUALITY = 80


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


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower() or mime_type.lower()


def _normalize_mime_type(mime_type: str) -> str | None:
    base = _base_mime_type(mime_type)
    if base == "image/png":
        return "image/png"
    if base in ("image/jpeg", "image/jpg"):
        return "image/jpeg"
    if base == "image/gif":
        return "image/gif"
    if base == "image/webp":
        return "image/webp"
    return None


def _encode_candidate(
    image: Image.Image, image_format: str, quality: int | None = None
) -> tuple[bytes, int]:
    buffer = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if quality is not None:
        save_kwargs["quality"] = quality
    if image_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image_format == "PNG" and image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        image = image.convert("RGBA" if has_alpha else "RGB")
    image.save(buffer, **save_kwargs)
    encoded = buffer.getvalue()
    return encoded, len(base64.b64encode(encoded))


def _dimension_note(original_size: tuple[int, int], size: tuple[int, int]) -> str:
    original_width, original_height = original_size
    width, height = size
    scale = original_width / width
    return (
        f"[Image: original {original_width}x{original_height}, displayed at {width}x{height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


def _conversion_hint(from_mime: str | None, to_mime: str) -> str | None:
    if not from_mime or from_mime == to_mime:
        return None
    return f"[Image converted from {from_mime} to {to_mime}.]"


def process_image_sync(
    data: bytes,
    mime_type: str | None = None,
    *,
    auto_resize: bool = True,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_base64_bytes: int = DEFAULT_MAX_BASE64_BYTES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict:
    """处理图片（对齐 TS image-process.ts / image-resize-core.ts）。

    PNG/JPEG/GIF/WebP 保留原格式；BMP 转 PNG。超过 4.5MB base64 或
    2000x2000 时用 PNG/JPEG 质量阶梯重编码。
    """
    if mime_type is None:
        mime_type = detect_supported_image_mime_type(data) or "image/png"
    original_mime = _base_mime_type(mime_type)
    normalized_mime = _normalize_mime_type(original_mime)

    try:
        converted_from: str | None = None
        normalized_bytes = data
        if normalized_mime is None:
            # 仅 BMP（及其他可解码格式）转 PNG。
            with Image.open(io.BytesIO(data)) as opened:
                image = ImageOps.exif_transpose(opened)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                normalized_bytes = buffer.getvalue()
            normalized_mime = "image/png"
            converted_from = original_mime

        with Image.open(io.BytesIO(normalized_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            original_size = image.size
            encoded_base64_size = len(base64.b64encode(normalized_bytes))

            # 已在尺寸与大小限制内：原样返回（不重编码）。
            if (
                image.size[0] <= max_dimension
                and image.size[1] <= max_dimension
                and encoded_base64_size < max_base64_bytes
            ):
                return {
                    "ok": True,
                    "data": normalized_bytes,
                    "mimeType": normalized_mime,
                    "hints": (
                        [_conversion_hint(converted_from, normalized_mime)]
                        if converted_from
                        else []
                    ),
                    "message": "",
                }

            if not auto_resize:
                return {
                    "ok": True,
                    "data": normalized_bytes,
                    "mimeType": normalized_mime,
                    "hints": (
                        [_conversion_hint(converted_from, normalized_mime)]
                        if converted_from
                        else []
                    ),
                    "message": "",
                }

            target_width, target_height = image.size
            if target_width > max_dimension:
                target_height = round(target_height * max_dimension / target_width)
                target_width = max_dimension
            if target_height > max_dimension:
                target_width = round(target_width * max_dimension / target_height)
                target_height = max_dimension

            quality_steps = list(dict.fromkeys([jpeg_quality, 85, 70, 55, 40]))
            current_width, current_height = target_width, target_height
            while True:
                resized = image.copy()
                resized.thumbnail((max(1, current_width), max(1, current_height)))
                candidates = [
                    _encode_candidate(resized, "PNG"),
                    *(_encode_candidate(resized, "JPEG", quality) for quality in quality_steps),
                ]
                for candidate_bytes, candidate_size in candidates:
                    if candidate_size < max_base64_bytes:
                        candidate_mime = (
                            "image/png" if candidate_bytes is candidates[0][0] else "image/jpeg"
                        )
                        hints: list[str] = []
                        conversion_hint = _conversion_hint(
                            converted_from or original_mime
                            if original_mime not in (normalized_mime, "image/png")
                            else None,
                            candidate_mime,
                        )
                        if conversion_hint:
                            hints.append(conversion_hint)
                        hints.append(_dimension_note(original_size, resized.size))
                        return {
                            "ok": True,
                            "data": candidate_bytes,
                            "mimeType": candidate_mime,
                            "hints": hints,
                            "message": "",
                        }

                if current_width <= 1 and current_height <= 1:
                    break
                next_width = 1 if current_width <= 1 else max(1, math.floor(current_width * 0.75))
                next_height = (
                    1 if current_height <= 1 else max(1, math.floor(current_height * 0.75))
                )
                if next_width == current_width and next_height == current_height:
                    break
                current_width, current_height = next_width, next_height

            return {
                "ok": False,
                "data": None,
                "mimeType": normalized_mime,
                "hints": [],
                "message": "[Image omitted: could not be resized below the inline image size limit.]",
            }
    except Exception:
        return {
            "ok": False,
            "data": None,
            "mimeType": original_mime or "image/png",
            "hints": [],
            "message": "[Image omitted: could not be converted to a supported inline image format.]",
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
