"""Judge 评分（对齐 TS vitest-evals judge 的最小移植）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

from .harness import JsonValue


@dataclass(slots=True)
class JudgeContext:
    """judge 收到的归一化运行数据。"""

    output: JsonValue
    events: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]


@dataclass(slots=True)
class JudgeResult:
    """单次 judge 评分结果。"""

    score: float
    metadata: dict[str, JsonValue] = field(default_factory=dict)


JudgeFn: TypeAlias = Callable[[JudgeContext], "JudgeResult | dict[str, Any] | float"]


def normalize_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 transcript 中的 tool_call / tool_result 事件合并为工具调用列表。

    每个条目包含 id / name / arguments / status（ok|error|pending），
    成功时带 result，失败时带 error（对齐 TS normalized toolCalls）。
    """
    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "tool_call":
            call_id = str(event.get("id") or "")
            records[call_id] = {
                "id": call_id,
                "name": event.get("name"),
                "arguments": event.get("arguments"),
                "status": "pending",
            }
            order.append(call_id)
        elif event_type == "tool_result":
            call_id = str(event.get("toolCallId") or "")
            record = records.get(call_id)
            if record is None:
                continue
            if bool(event.get("isError")):
                record["status"] = "error"
                record["error"] = event.get("error") or event.get("content") or "Tool failed"
            else:
                record["status"] = "ok"
                record["result"] = event.get("content")
    return [records[call_id] for call_id in order]


class Judge:
    """命名评分器：把运行数据转换为 0..1 分数与元数据。"""

    def __init__(self, name: str, fn: JudgeFn) -> None:
        stripped = name.strip()
        if not stripped:
            raise ValueError("Judge name must not be empty.")
        self.name = stripped
        self._fn = fn

    def evaluate(self, context: JudgeContext) -> JudgeResult:
        result = self._fn(context)
        if isinstance(result, (int, float)):
            return JudgeResult(score=float(result))
        if isinstance(result, dict):
            score = result.get("score", 0)
            metadata = result.get("metadata") or {}
            return JudgeResult(
                score=float(score) if isinstance(score, (int, float)) else 0.0,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        raise TypeError("Judge function must return a number or a {score, metadata} dict.")


def create_judge(name: str, fn: JudgeFn) -> Judge:
    """创建命名 judge（对齐 TS createJudge）。"""
    return Judge(name, fn)


def average_judge_scores(judges: list[Judge], context: JudgeContext) -> float | None:
    """计算一组 judge 的平均分；无 judge 时返回 None。"""
    if not judges:
        return None
    scores = [judge.evaluate(context).score for judge in judges]
    return sum(scores) / len(scores)


__all__ = [
    "Judge",
    "JudgeContext",
    "JudgeResult",
    "average_judge_scores",
    "create_judge",
    "normalize_tool_calls",
]
