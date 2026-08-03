"""JSONL 帧编解码（对齐 TS modes/rpc/jsonl.ts）。

帧分割仅按 `\\n`（严格 JSONL）；payload 内的 U+2028/U+2029 不参与分割。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator


def _json_default(value: Any) -> Any:
    """dataclass → dict（事件中的 Model 等对象可序列化）。"""
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_json_line(value: Any) -> str:
    """序列化单条严格 JSONL 记录（LF 结尾）。"""
    return json.dumps(value, ensure_ascii=False, default=_json_default) + "\n"


async def read_jsonl_lines(reader) -> AsyncIterator[str]:
    """从异步流读取 JSONL 行（只按 \\n 分割；容忍 \\r 结尾）。"""
    while True:
        line = await reader.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace")
        if text.endswith("\n"):
            text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
        yield text


__all__ = ["serialize_json_line", "read_jsonl_lines"]
