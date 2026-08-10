"""JSON 可序列化校验（对齐 TS `session.ts` 的 assertJsonSerializable）。

写入 mutation 前校验，防止循环引用、NaN、非 JSON 类型进入会话日志。
"""

from __future__ import annotations

import math
from typing import Any

from .types import SessionError


def _invalid_payload(reason: str) -> None:
    raise SessionError("invalid_payload", f"Durable payload {reason}")


def assert_json_serializable(value: Any) -> None:
    """校验 value 可无损序列化为 JSON；失败抛 SessionError(invalid_payload)。"""
    active: set[int] = set()
    stack: list[tuple[Any, bool]] = [(value, False)]
    while stack:
        candidate, exiting = stack.pop()
        if exiting:
            active.discard(id(candidate))
            continue
        if candidate is None or isinstance(candidate, (bool, int, str)):
            continue
        if isinstance(candidate, float):
            if not math.isfinite(candidate):
                _invalid_payload("contains a non-finite number")
            continue
        if isinstance(candidate, list):
            if id(candidate) in active:
                _invalid_payload("contains a cycle")
            active.add(id(candidate))
            stack.append((candidate, True))
            stack.extend((item, False) for item in candidate)
            continue
        if isinstance(candidate, dict):
            if id(candidate) in active:
                _invalid_payload("contains a cycle")
            active.add(id(candidate))
            stack.append((candidate, True))
            for key, item in candidate.items():
                if not isinstance(key, str):
                    _invalid_payload("contains a non-string key")
                stack.append((item, False))
            continue
        _invalid_payload(f"contains {type(candidate).__name__}")


__all__ = ["assert_json_serializable"]
