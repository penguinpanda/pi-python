"""工具路径解析（对齐 TS harness/tools/path-utils.ts）。"""

from __future__ import annotations

import os
import re
import unicodedata

from ..env import ExecutionEnv, FileError, get_or_throw

_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"


def is_path_within(path: str, root: str) -> bool:
    """path 是否位于 root 内（realpath 后按公共路径比较，跨盘/不存在返回 False）。"""
    try:
        root_real = os.path.realpath(root)
        path_real = os.path.realpath(path)
        return os.path.commonpath([root_real, path_real]) == root_real
    except (OSError, ValueError):
        return False


def _normalize_tool_path(path: str) -> str:
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, signal=None) -> str:
    result = await env.absolute_path(_normalize_tool_path(path), signal)
    resolved = get_or_throw(result)
    if getattr(env, "restrict_paths_to_cwd", False) and not is_path_within(resolved, env.cwd):
        raise FileError(
            "permission_denied",
            f"Path is outside the working directory: {path}",
            resolved,
        )
    return resolved


async def resolve_read_tool_path(env: ExecutionEnv, path: str, signal=None) -> str:
    # read 工具允许全盘读取（对齐 TS）；仅当调用方显式开启限制时校验。
    resolved = await resolve_tool_path(env, path, signal)
    variants = [
        resolved,
        re.sub(
            r" (AM|PM)\.", lambda m: f"{_NARROW_NO_BREAK_SPACE}{m.group(1)}.", resolved, flags=re.I
        ),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ]
    for variant in dict.fromkeys(variants):
        exists = await env.exists(variant, signal)
        if exists[0] and exists[1]:
            return variant
    return resolved
