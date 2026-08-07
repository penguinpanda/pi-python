"""Eval Reporter（对齐 TS vitest-evals/reporter.ts 的观测收集与持久化）。

将观测收集、报告生成抽离为独立模块，
便于替换报告格式（JSON / 终端 / CI）而不修改 runner。
"""

from __future__ import annotations

from collections.abc import Sequence

from .harness_table import (
    EVAL_HARNESS_ITERATION_ARTIFACT,
    parse_eval_harness_iteration_artifact,
)
from .summary import (
    HarnessObservation,
    format_harness_comparison_report,
    summarize_harness_comparisons,
)


def read_finite_number(value: object) -> int | float | None:
    """读取有限数值；非数字返回 None。"""
    if (
        isinstance(value, (int, float))
        and value == value
        and value not in (float("inf"), float("-inf"))
    ):
        return value
    return None


def _make_observation(
    *,
    eval_set: str,
    group_key: str,
    test_name: str,
    file: str,
    harness: str,
    baseline: str,
    candidates: list[str],
    repetition: int,
    outcome: str,
    score: float | None = None,
    total_tokens: int | None = None,
    total_ms: float | None = None,
    estimated_cost_usd: float | None = None,
) -> HarnessObservation:
    """构造 HarnessObservation；outcome 必填且仅 scored 时传入 score。"""
    return HarnessObservation(
        eval_set=eval_set,
        group_key=group_key,
        test_name=test_name,
        file=file,
        harness=harness,
        baseline=baseline,
        candidates=candidates,
        repetition=repetition,
        outcome=outcome,  # type: ignore[arg-type]
        score=score,  # type: ignore[arg-type]
        total_tokens=total_tokens,
        total_ms=total_ms,
        estimated_cost_usd=estimated_cost_usd,
    )


def collect_observations(
    runs: Sequence[tuple[str, "CaseResult | None"]],
) -> list[HarnessObservation]:
    """从多个 CaseResult 收集可对比的观测列表。

    每个元组为 (relative_module_id, CaseResult | None)。
    """
    observations: list[HarnessObservation] = []
    for file, result in runs:
        if result is None:
            continue
        run = result.run
        if run is None:
            continue
        iteration = parse_eval_harness_iteration_artifact(
            run.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT)
        )
        if iteration is None:
            continue
        metadata = run.usage.get("metadata") or {}
        total_tokens_val = read_finite_number(run.usage.get("totalTokens"))
        total_ms_val = read_finite_number(run.timings.get("totalMs"))
        estimated_cost_usd_val = read_finite_number(metadata.get("estimatedCostUsd"))
        kwargs: dict[str, object] = {
            "eval_set": iteration.eval_set,
            "group_key": iteration.group_key,
            "test_name": result.case.name,
            "file": file,
            "harness": iteration.harness,
            "baseline": iteration.baseline,
            "candidates": iteration.candidates,
            "repetition": iteration.repetition,
            "total_tokens": int(total_tokens_val) if total_tokens_val is not None else None,
            "total_ms": float(total_ms_val) if total_ms_val is not None else None,
            "estimated_cost_usd": (
                float(estimated_cost_usd_val) if estimated_cost_usd_val is not None else None
            ),
        }
        if run.errors:
            observations.append(_make_observation(**kwargs, outcome="errored"))  # type: ignore[arg-type]
        elif result.avg_score is not None:
            observations.append(
                _make_observation(**kwargs, outcome="scored", score=result.avg_score)  # type: ignore[arg-type]
            )
        elif result.failed:
            observations.append(_make_observation(**kwargs, outcome="errored"))  # type: ignore[arg-type]
        else:
            observations.append(_make_observation(**kwargs, outcome="unscored"))  # type: ignore[arg-type]
    return observations


def generate_report(observations: list[HarnessObservation]) -> str:
    """生成终端可读的对比报告。"""
    summary = summarize_harness_comparisons(observations)
    return format_harness_comparison_report(summary)


# 为了类型检查，需要从 suite 导入 CaseResult
from .suite import CaseResult  # noqa: E402  # isort:skip
