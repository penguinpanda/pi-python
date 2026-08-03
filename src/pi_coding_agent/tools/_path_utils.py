"""coding-agent 特有工具（grep/find/ls）的路径解析辅助。"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_cwd_path(base: Path, file_path: str) -> Path:
    """解析并安全检查文件路径（必须位于 cwd 下）。"""
    p = (base / file_path).resolve()
    # 确保在 cwd 下
    if not str(p).startswith(str(base) + os.sep) and p != base:
        raise ValueError(f"Path traversal detected: {file_path}")
    return p


__all__ = ["resolve_cwd_path"]
