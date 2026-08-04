"""工具路径解析（对齐 TS harness/tools/path-utils.ts）。"""

from __future__ import annotations

import re
import unicodedata

from ..env import ExecutionEnv, get_or_throw

_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"


def _normalize_tool_path(path: str) -> str:
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, signal=None) -> str:
    result = await env.absolute_path(_normalize_tool_path(path), signal)
    return get_or_throw(result)


async def resolve_read_tool_path(env: ExecutionEnv, path: str, signal=None) -> str:
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
