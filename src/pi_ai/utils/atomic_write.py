"""原子 JSON 文件写入（凭证 / 模型目录共用）。

- 随机后缀临时文件 + O_CREAT|O_EXCL：固定名可预测、符号链接攻击均不可行；
- 创建即按 ``mode``（默认 0600）收紧权限，写入前不暴露明文；
- 写入成功后 ``os.replace`` 原子替换目标；
- 任一环节失败时清理临时文件，不残留。
"""

from __future__ import annotations

import json
import os
import secrets

from pathlib import Path
from typing import Any


class AtomicWriteError(OSError):
    """无法创建临时文件（多次随机名冲突）。"""


def atomic_write_json(path: str | Path, data: Any, *, mode: int = 0o600) -> None:
    """把 ``data`` 原子写入 ``path``（JSON，indent=2，ensure_ascii=False）。"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for _ in range(3):
        tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError as exc:
            last_error = exc
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
            os.replace(tmp, dest)
            return
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    raise AtomicWriteError(f"Failed to create temp file for {dest}") from last_error


__all__ = ["AtomicWriteError", "atomic_write_json"]
