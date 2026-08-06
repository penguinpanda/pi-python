"""Harness 对比汇总与报告（对齐 TS vitest-evals/summary.ts 移植）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

HarnessObservationOutcome: TypeAlias = Literal[
    "scored", "unscored", "skipped", "pending", "errored"
]
HarnessDiagnosticReason: TypeAlias = Literal[
    "missing-observation",
    "duplicate-observation",
    "harness-error",
    "missing-score",
    "unscorable-outcome",
]


@dataclass(slots=True)
class HarnessObservation:
    """一次可对比的 harness 运行观测。"""

    eval_set: str
    group_key: str
    test_name: str
    file: str
    harness: str
    baseline: str
    candidates: list[str]
    repetition: int
    outcome: HarnessObservationOutcome
    score: float | None = None
    total_tokens: int | None = None
    total_ms: float | None = None
    estimated_cost_usd: float | None = None


_MetricSelector: TypeAlias = Callable[["HarnessObservation"], int | float | None]


@dataclass(slots=True)
class PairedMetricSummary:
    """配对指标的均值与差值。"""

    total_pairs: int
    eligible_pairs: int
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    mean_delta: float | None = None


@dataclass(slots=True)
class CorrectnessLiftSummary:
    """正确率 lift（candidate - baseline，百分点）。"""

    total_pairs: int
    eligible_pairs: int
    baseline_pass_rate: float | None = None
    candidate_pass_rate: float | None = None
    lift: float | None = None
    baseline_wins: int = 0
    candidate_wins: int = 0
    ties: int = 0


@dataclass(slots=True)
class HarnessPairComparison:
    """baseline 与一个 candidate 的完整对比。"""

    baseline: str
    candidate: str
    correctness: CorrectnessLiftSummary
    total_tokens: PairedMetricSummary
    total_ms: PairedMetricSummary
    estimated_cost_usd: PairedMetricSummary


@dataclass(slots=True)
class HarnessComparisonDiagnostic:
    """无法配对的观测诊断。"""

    eval_set: str
    group_key: str
    test_name: str
    file: str
    repetition: int
    harness: str
    reason: HarnessDiagnosticReason


@dataclass(slots=True)
class HarnessEvalSetReport:
    """单个 eval set 的对比报告。"""

    eval_set: str
    comparisons: list[HarnessPairComparison] = field(default_factory=list)


@dataclass(slots=True)
class HarnessComparisonReport:
    """全量对比报告。"""

    schema_version: int = 1
    eval_sets: list[HarnessEvalSetReport] = field(default_factory=list)
    diagnostics: list[HarnessComparisonDiagnostic] = field(default_factory=list)


@dataclass(slots=True)
class _HarnessDescriptor:
    name: str
    index: int


@dataclass(slots=True)
class _ObservationGroup:
    eval_set: str
    group_key: str
    test_name: str
    file: str
    repetition: int
    observations_by_harness: dict[str, list[HarnessObservation]] = field(default_factory=dict)


@dataclass(slots=True)
class _EvalSetData:
    baseline: _HarnessDescriptor
    candidates_by_name: dict[str, _HarnessDescriptor] = field(default_factory=dict)
    groups_by_key: dict[str, _ObservationGroup] = field(default_factory=dict)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _precise_difference(left: float, right: float) -> float:
    return float(f"{left - right:.15g}")


def _group_observations(
    observations: list[HarnessObservation],
) -> dict[str, _EvalSetData]:
    eval_sets: dict[str, _EvalSetData] = {}
    for observation in observations:
        data = eval_sets.setdefault(
            observation.eval_set,
            _EvalSetData(baseline=_HarnessDescriptor(observation.baseline, 0)),
        )
        for index, name in enumerate(observation.candidates):
            existing = data.candidates_by_name.get(name)
            if existing is None or index < existing.index:
                data.candidates_by_name[name] = _HarnessDescriptor(name, index)
        group_key = observation.group_key
        group = data.groups_by_key.setdefault(
            group_key,
            _ObservationGroup(
                eval_set=observation.eval_set,
                group_key=group_key,
                test_name=observation.test_name,
                file=observation.file,
                repetition=observation.repetition,
            ),
        )
        group.observations_by_harness.setdefault(observation.harness, []).append(observation)
    return eval_sets


def _ordered_harnesses(data: _EvalSetData) -> list[_HarnessDescriptor]:
    return [
        data.baseline,
        *sorted(
            data.candidates_by_name.values(),
            key=lambda item: (item.index, item.name),
        ),
    ]


def _ordered_candidates(data: _EvalSetData) -> list[_HarnessDescriptor]:
    return sorted(data.candidates_by_name.values(), key=lambda item: (item.index, item.name))


def _ordered_groups(data: _EvalSetData) -> list[_ObservationGroup]:
    return sorted(
        data.groups_by_key.values(),
        key=lambda group: (group.group_key, group.repetition),
    )


def _collect_diagnostics(
    harnesses: list[_HarnessDescriptor],
    groups: list[_ObservationGroup],
) -> list[HarnessComparisonDiagnostic]:
    diagnostics: list[HarnessComparisonDiagnostic] = []
    for group in groups:
        for harness in harnesses:
            observations = group.observations_by_harness.get(harness.name, [])
            reason: HarnessDiagnosticReason | None = None
            if not observations:
                reason = "missing-observation"
            elif len(observations) > 1:
                reason = "duplicate-observation"
            elif observations[0].outcome == "errored":
                reason = "harness-error"
            elif observations[0].outcome == "unscored":
                reason = "missing-score"
            elif observations[0].outcome != "scored":
                reason = "unscorable-outcome"
            if reason is None:
                continue
            diagnostics.append(
                HarnessComparisonDiagnostic(
                    eval_set=group.eval_set,
                    group_key=group.group_key,
                    test_name=group.test_name,
                    file=group.file,
                    repetition=group.repetition,
                    harness=harness.name,
                    reason=reason,
                )
            )
    return diagnostics


def _pair_observations(
    groups: list[_ObservationGroup],
    baseline_harness: str,
    candidate_harness: str,
) -> list[tuple[HarnessObservation, HarnessObservation]]:
    pairs: list[tuple[HarnessObservation, HarnessObservation]] = []
    for group in groups:
        baseline = group.observations_by_harness.get(baseline_harness, [])
        candidate = group.observations_by_harness.get(candidate_harness, [])
        if len(baseline) == 1 and len(candidate) == 1:
            pairs.append((baseline[0], candidate[0]))
    return pairs


def _summarize_metric(
    pairs: list[tuple[HarnessObservation, HarnessObservation]],
    select: _MetricSelector,
    total_pairs: int,
) -> PairedMetricSummary:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for baseline, candidate in pairs:
        if baseline.outcome != "scored" or candidate.outcome != "scored":
            continue
        baseline_value = select(baseline)
        candidate_value = select(candidate)
        if not isinstance(baseline_value, (int, float)) or not isinstance(
            candidate_value, (int, float)
        ):
            continue
        baseline_values.append(float(baseline_value))
        candidate_values.append(float(candidate_value))
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    return PairedMetricSummary(
        total_pairs=total_pairs,
        eligible_pairs=len(baseline_values),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=(
            _precise_difference(candidate_mean, baseline_mean)
            if baseline_mean is not None and candidate_mean is not None
            else None
        ),
    )


def _summarize_correctness(
    pairs: list[tuple[HarnessObservation, HarnessObservation]],
    total_pairs: int,
) -> CorrectnessLiftSummary:
    eligible_pairs = 0
    baseline_passes = 0
    candidate_passes = 0
    baseline_wins = 0
    candidate_wins = 0
    ties = 0
    for baseline, candidate in pairs:
        if baseline.outcome != "scored" or candidate.outcome != "scored":
            continue
        eligible_pairs += 1
        baseline_passed = (baseline.score or 0) >= 1
        candidate_passed = (candidate.score or 0) >= 1
        if baseline_passed:
            baseline_passes += 1
        if candidate_passed:
            candidate_passes += 1
        if baseline_passed == candidate_passed:
            ties += 1
        elif baseline_passed:
            baseline_wins += 1
        else:
            candidate_wins += 1
    baseline_pass_rate = baseline_passes / eligible_pairs if eligible_pairs else None
    candidate_pass_rate = candidate_passes / eligible_pairs if eligible_pairs else None
    return CorrectnessLiftSummary(
        total_pairs=total_pairs,
        eligible_pairs=eligible_pairs,
        baseline_pass_rate=baseline_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        lift=(
            _precise_difference(candidate_pass_rate, baseline_pass_rate)
            if baseline_pass_rate is not None and candidate_pass_rate is not None
            else None
        ),
        baseline_wins=baseline_wins,
        candidate_wins=candidate_wins,
        ties=ties,
    )


def _compare_harnesses(
    baseline: _HarnessDescriptor,
    candidate: _HarnessDescriptor,
    groups: list[_ObservationGroup],
) -> HarnessPairComparison:
    pairs = _pair_observations(groups, baseline.name, candidate.name)
    return HarnessPairComparison(
        baseline=baseline.name,
        candidate=candidate.name,
        correctness=_summarize_correctness(pairs, len(groups)),
        total_tokens=_summarize_metric(
            pairs,
            lambda observation: observation.total_tokens,
            len(groups),
        ),
        total_ms=_summarize_metric(
            pairs,
            lambda observation: observation.total_ms,
            len(groups),
        ),
        estimated_cost_usd=_summarize_metric(
            pairs,
            lambda observation: observation.estimated_cost_usd,
            len(groups),
        ),
    )


def summarize_harness_comparisons(
    observations: list[HarnessObservation],
) -> HarnessComparisonReport:
    """按 eval set 汇总对比（对齐 TS summarizeHarnessComparisons）。"""
    eval_sets: list[HarnessEvalSetReport] = []
    diagnostics: list[HarnessComparisonDiagnostic] = []
    for eval_set, data in sorted(
        _group_observations(observations).items(),
        key=lambda item: item[0],
    ):
        harnesses = _ordered_harnesses(data)
        candidates = _ordered_candidates(data)
        groups = _ordered_groups(data)
        eval_sets.append(
            HarnessEvalSetReport(
                eval_set=eval_set,
                comparisons=[
                    _compare_harnesses(data.baseline, candidate, groups) for candidate in candidates
                ],
            )
        )
        diagnostics.extend(_collect_diagnostics(harnesses, groups))
    diagnostics.sort(
        key=lambda item: (
            item.eval_set,
            item.file,
            item.group_key,
            item.repetition,
            item.harness,
        )
    )
    return HarnessComparisonReport(eval_sets=eval_sets, diagnostics=diagnostics)


def _format_percentage(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:.1f}%"


def _format_signed(value: float, fraction_digits: int) -> str:
    return f"{value:+.{fraction_digits}f}"


def _format_coverage(eligible_pairs: int, total_pairs: int) -> str:
    return f"({eligible_pairs}/{total_pairs} pairs)"


def _format_metric(
    label: str,
    metric: PairedMetricSummary,
    format_value: object,
    format_delta: object,
    comparison_pairs: int,
) -> str:
    value_formatter = format_value if callable(format_value) else lambda value: str(value)
    delta_formatter = format_delta if callable(format_delta) else lambda value: str(value)
    coverage = ""
    if metric.eligible_pairs not in (0, comparison_pairs):
        coverage = f" {_format_coverage(metric.eligible_pairs, metric.total_pairs)}"
    if metric.baseline_mean is None or metric.candidate_mean is None or metric.mean_delta is None:
        return f"    {label:<9}  unavailable{coverage}"
    delta = delta_formatter(metric.mean_delta)
    values = (
        f"(candidate {value_formatter(metric.candidate_mean)}, "
        f"baseline {value_formatter(metric.baseline_mean)})"
    )
    return f"    {label:<9}  {delta} {values}{coverage}"


def format_harness_comparison_report(report: HarnessComparisonReport) -> str:
    """把对比报告渲染为文本（对齐 TS formatHarnessComparisonReport）。"""
    if all(not eval_set.comparisons for eval_set in report.eval_sets):
        return ""
    lines = ["Eval Comparisons"]
    for eval_set in report.eval_sets:
        lines.append(f"  {eval_set.eval_set}")
        for index, comparison in enumerate(eval_set.comparisons):
            if index > 0:
                lines.append("")
            correctness = comparison.correctness
            lines.append(f"    {'Baseline':<9}  {comparison.baseline}")
            lines.append(
                f"    {'Candidate':<9}  {comparison.candidate} "
                f"{_format_coverage(correctness.eligible_pairs, correctness.total_pairs)}"
            )
            if correctness.lift is None:
                lines.append(f"    {'Pass rate':<9}  unavailable")
            else:
                lift = correctness.lift * 100
                lines.append(
                    f"    {'Pass rate':<9}  {_format_signed(lift, 1)} pp "
                    f"(candidate {_format_percentage(correctness.candidate_pass_rate)}, "
                    f"baseline {_format_percentage(correctness.baseline_pass_rate)})"
                )
            lines.append(
                _format_metric(
                    "Tokens",
                    comparison.total_tokens,
                    lambda value: f"{value:.1f}",
                    lambda value: _format_signed(value, 1),
                    correctness.eligible_pairs,
                )
            )
            lines.append(
                _format_metric(
                    "Latency",
                    comparison.total_ms,
                    lambda value: f"{value:.1f}ms",
                    lambda value: f"{_format_signed(value, 1)}ms",
                    correctness.eligible_pairs,
                )
            )
            lines.append(
                _format_metric(
                    "Est. cost",
                    comparison.estimated_cost_usd,
                    lambda value: f"${value:.4f}",
                    lambda value: f"+${abs(value):.4f}" if value >= 0 else f"-${abs(value):.4f}",
                    correctness.eligible_pairs,
                )
            )
    if report.diagnostics:
        lines.append("  Incomplete observations")
        for diagnostic in report.diagnostics:
            lines.append(
                f"    {diagnostic.reason}: {diagnostic.file}/{diagnostic.test_name} "
                f"repetition {diagnostic.repetition}, harness {diagnostic.harness}"
            )
    return "\n".join(lines)


__all__ = [
    "CorrectnessLiftSummary",
    "HarnessComparisonDiagnostic",
    "HarnessComparisonReport",
    "HarnessEvalSetReport",
    "HarnessObservation",
    "HarnessPairComparison",
    "PairedMetricSummary",
    "format_harness_comparison_report",
    "summarize_harness_comparisons",
]
