"""通用评测 harness 类型（对齐 TS vitest-evals/harness 的最小移植）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
EvalInput: TypeAlias = JsonValue


@dataclass(slots=True)
class HarnessContext:
    """一次 eval run 的上下文：harness 可在此写入 artifacts。"""

    artifacts: dict[str, JsonValue] = field(default_factory=dict)

    def set_artifact(self, name: str, value: JsonValue) -> None:
        self.artifacts[name] = value


@dataclass(slots=True)
class HarnessRun:
    """一次 eval run 的结果（对齐 vitest-evals HarnessRun）。"""

    output: JsonValue
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, JsonValue] = field(default_factory=dict)


class Harness(Protocol):
    """评测 harness：对同一类输入运行被测系统。"""

    name: str

    async def run(self, input: EvalInput, context: HarnessContext) -> HarnessRun: ...


@dataclass(slots=True)
class FunctionHarness:
    """由单个异步函数构造的 harness。"""

    name: str
    run_fn: Callable[[EvalInput, HarnessContext], Awaitable[HarnessRun]]

    async def run(self, input: EvalInput, context: HarnessContext) -> HarnessRun:
        return await self.run_fn(input, context)


def create_harness(
    name: str,
    run_fn: Callable[[EvalInput, HarnessContext], Awaitable[HarnessRun]],
) -> FunctionHarness:
    """创建命名 harness（对齐 TS createHarness）。"""
    stripped = name.strip()
    if not stripped:
        raise ValueError("Harness name must not be empty.")
    return FunctionHarness(name=stripped, run_fn=run_fn)


def canonicalize_json(value: Any, ancestors: set[int] | None = None) -> JsonValue:
    """把 eval 输入规范化为稳定 JSON（对齐 TS canonicalizeJson）。

    拒绝非有限数字、非 JSON 类型与循环引用；对象键按字典序排序。
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Eval input must contain only finite numbers.")
        return value
    if not isinstance(value, (list, dict)):
        raise TypeError("Eval input must be JSON-serializable.")
    seen = ancestors if ancestors is not None else set()
    marker = id(value)
    if marker in seen:
        raise TypeError("Eval input must not contain circular references.")
    seen.add(marker)
    try:
        if isinstance(value, list):
            return [canonicalize_json(item, seen) for item in value]
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Eval input object keys must be strings.")
        return {key: canonicalize_json(item, seen) for key, item in sorted(value.items())}
    finally:
        seen.remove(marker)


__all__ = [
    "EvalInput",
    "FunctionHarness",
    "Harness",
    "HarnessContext",
    "HarnessRun",
    "JsonValue",
    "canonicalize_json",
    "create_harness",
]
