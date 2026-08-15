"""@file 参数处理（对齐 TS cli/file-processor.ts）。"""

from __future__ import annotations

import base64
from pathlib import Path

from pi_ai import ImageContent

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# 文本注入大小上限（防止大文件/二进制被打进上下文）。
MAX_TEXT_FILE_BYTES = 1_000_000
MAX_IMAGE_FILE_BYTES = 20_000_000
# 目录递归时忽略的常见目录名。
_IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}
# 已知二进制/压缩扩展：不做文本注入（未知扩展由内容嗅探兜底）。
_KNOWN_BINARY_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".ico",
    ".bin",
    ".wasm",
    ".pyc",
    ".class",
    ".jar",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}
_TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".sh",
    ".bat",
    ".ps1",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rb",
    ".php",
    ".sql",
    ".xml",
    ".ini",
    ".cfg",
    ".env",
}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _mime_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "application/octet-stream")


def _should_inject_text(path: Path) -> bool:
    # 已知二进制/压缩扩展直接跳过；未知扩展由内容嗅探兜底。
    return path.suffix.lower() not in _KNOWN_BINARY_EXTENSIONS


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _inject_file_text(path: Path, data: bytes) -> str:
    return f"### File: {path}\n\n{data.decode('utf-8', errors='replace')}"


def _process_path(
    path: Path,
    texts: list[str],
    images: list[ImageContent],
    *,
    max_text_bytes: int = MAX_TEXT_FILE_BYTES,
    max_image_bytes: int = MAX_IMAGE_FILE_BYTES,
) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if any(part in _IGNORED_DIR_NAMES for part in child.parts):
                continue
            if child.is_file():
                _process_path(
                    child,
                    texts,
                    images,
                    max_text_bytes=max_text_bytes,
                    max_image_bytes=max_image_bytes,
                )
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if is_image_path(path):
        if size > max_image_bytes:
            return
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return
        images.append(
            ImageContent(
                type="image",
                url=None,
                data=data,
                mime_type=_mime_type(path),
            )
        )
        return
    if not _should_inject_text(path):
        return
    if size > max_text_bytes:
        return
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return
    if _is_binary(raw_bytes):
        return
    texts.append(_inject_file_text(path, raw_bytes))


async def process_at_files(
    args: list[str],
    cwd: str,
) -> tuple[list[str], list[ImageContent]]:
    """处理 `@path` 参数：文本注入 / 图片转 ImageContent / 目录递归展开。"""
    texts: list[str] = []
    images: list[ImageContent] = []
    base = Path(cwd).expanduser()
    for arg in args:
        if not arg.startswith("@"):
            texts.append(arg)
            continue
        raw = arg[1:]
        path = Path(raw)
        if not path.is_absolute():
            path = base / raw
        path = path.resolve()
        if not path.exists():
            # 不存在的 @path 保留原文。
            texts.append(arg)
            continue
        _process_path(path, texts, images)
    return texts, images


__all__ = [
    "IMAGE_EXTENSIONS",
    "is_image_path",
    "process_at_files",
]


# ---------------------------------------------------------------------------
# TS cli/file-processor.ts 兼容入口
# ---------------------------------------------------------------------------

_SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}
_MAX_IMAGE_DIMENSION = 2000


def _sniff_image_mime_type(data: bytes) -> str | None:
    """Sniff supported inline image MIME type from file content."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_format_for_mime(mime_type: str) -> str:
    return _SUPPORTED_IMAGE_MIME_TYPES.get(mime_type, "PNG")


def _resize_image_bytes(data: bytes, mime_type: str) -> tuple[bytes, str, bool]:
    """Resize oversized images and normalize unsupported formats to PNG."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(data))
    image.load()
    resized = False
    if max(image.size) > _MAX_IMAGE_DIMENSION:
        image.thumbnail((_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION))
        resized = True
    out_mime = mime_type if mime_type in _SUPPORTED_IMAGE_MIME_TYPES else "image/png"
    out_format = _image_format_for_mime(out_mime)
    buffer = BytesIO()
    save_options = {}
    if out_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    try:
        image.save(buffer, format=out_format, **save_options)
    except (KeyError, ValueError):
        # Some Pillow builds cannot encode WEBP; fall back to PNG.
        out_mime = "image/png"
        image.save(buffer, format="PNG")
    return buffer.getvalue(), out_mime, resized


async def process_file_arguments(
    file_args: list[str],
    cwd: str,
    *,
    auto_resize_images: bool = True,
) -> tuple[str, list[ImageContent]]:
    """Process TS-style ``@file`` arguments.

    Returns one combined ``<file>...</file>`` text block and image attachments.
    Missing files, unreadable files, and directories raise ``OSError`` (the TS
    CLI prints the error and exits with status 1).
    """
    text = ""
    images: list[ImageContent] = []
    base = Path(cwd).expanduser().resolve()

    for file_arg in file_args:
        raw = file_arg[1:] if file_arg.startswith("@") else file_arg
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            raise IsADirectoryError(str(path))
        size = path.stat().st_size
        if size == 0:
            continue
        raw_bytes = path.read_bytes()
        mime_type = _sniff_image_mime_type(raw_bytes)

        if mime_type is not None:
            image_data = base64.b64encode(raw_bytes).decode("ascii")
            hints: list[str] = []
            if auto_resize_images:
                try:
                    resized_bytes, out_mime, resized = _resize_image_bytes(raw_bytes, mime_type)
                    image_data = base64.b64encode(resized_bytes).decode("ascii")
                    mime_type = out_mime
                    if resized:
                        from PIL import Image
                        from io import BytesIO

                        width, height = Image.open(BytesIO(resized_bytes)).size
                        hints.append(
                            f"[Image resized from its original dimensions to {width}x{height}.]"
                        )
                except Exception:
                    # Keep the original bytes when resize/convert fails.
                    pass
            images.append(
                ImageContent(
                    type="image",
                    url=None,
                    data=image_data,
                    mime_type=mime_type,
                )
            )
            text += (
                f'<file name="{path}">'
                + ("\n".join(hints) if hints else "")
                + "</file>\n"
            )
            continue

        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not read file {path}: {exc}") from exc
        text += f'<file name="{path}">\n{content}\n</file>\n'

    return text, images
