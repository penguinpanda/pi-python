"""@file 参数处理（对齐 TS cli/file-processor.ts）。"""

from __future__ import annotations

import base64
from pathlib import Path

from pi_ai import ImageContent

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
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
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    # 未知扩展：尝试按文本读取，失败则跳过。
    return True


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None


def _inject_file_text(path: Path) -> str | None:
    content = _read_text(path)
    if content is None:
        return None
    return f"### File: {path}\n\n{content}"


def _process_path(path: Path, texts: list[str], images: list[ImageContent]) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                _process_path(child, texts, images)
        return
    if is_image_path(path):
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
    injected = _inject_file_text(path)
    if injected is not None:
        texts.append(injected)


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
