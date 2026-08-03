"""图片 MIME 检测与 Base64 编码（对齐 TS harness/tools/image.ts）。"""

from __future__ import annotations

import base64

_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def _read_u32be(data: bytes, offset: int) -> int:
    return (
        data[offset] * 0x1000000
        + (data[offset + 1] << 16)
        + (data[offset + 2] << 8)
        + data[offset + 3]
    )


def _read_u32le(data: bytes, offset: int) -> int:
    return (
        data[offset]
        + (data[offset + 1] << 8)
        + (data[offset + 2] << 16)
        + (data[offset + 3] * 0x1000000)
    )


def _is_png(data: bytes) -> bool:
    return (
        len(data) >= 16
        and _read_u32be(data, len(_PNG_SIGNATURE)) == 13
        and data[12:16] == b"IHDR"
    )


def _is_animated_png(data: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        chunk_length = _read_u32be(data, offset)
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"acTL":
            return True
        if chunk_type == b"IDAT":
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(data):
            return False
        offset = next_offset
    return False


def _is_bmp(data: bytes) -> bool:
    if len(data) < 26:
        return False
    declared_size = _read_u32le(data, 2)
    pixel_offset = _read_u32le(data, 10)
    dib_size = _read_u32le(data, 14)
    if declared_size != 0 and declared_size < 26:
        return False
    if pixel_offset < 14 + dib_size:
        return False
    if declared_size != 0 and pixel_offset >= declared_size:
        return False
    if dib_size == 12:
        planes = int.from_bytes(data[22:24], "little")
        bpp = int.from_bytes(data[24:26], "little")
    elif 40 <= dib_size <= 124:
        if len(data) < 30:
            return False
        planes = int.from_bytes(data[26:28], "little")
        bpp = int.from_bytes(data[28:30], "little")
    else:
        return False
    return planes == 1 and bpp in (1, 4, 8, 16, 24, 32)


def detect_supported_image_mime_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return None if len(data) > 3 and data[3] == 0xF7 else "image/jpeg"
    if data.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(data) and not _is_animated_png(data) else None
    if data.startswith(b"GIF"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM") and _is_bmp(data):
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
