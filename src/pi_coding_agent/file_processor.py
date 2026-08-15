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
