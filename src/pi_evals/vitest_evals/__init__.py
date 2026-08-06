"""vitest-evals 的 Python 最小移植：harness / judge / 对比表 / artifacts / summary。"""

from __future__ import annotations

from .artifacts import (
    PI_EVAL_SOURCES_ARTIFACT,
    PI_SESSION_SNAPSHOT_ARTIFACT,
    persist_eval_artifact_references,
)
from .harness import (
    EvalInput,
    FunctionHarness,
    Harness,
    HarnessContext,
    HarnessRun,
    JsonValue,
    canonicalize_json,
    create_harness,
)
from .harness_table import (
    EVAL_HARNESS_ITERATION_ARTIFACT,
    EvalHarnessRow,
    derive_eval_group_key,
    derive_input_key,
    eval_harness_table,
    parse_eval_harness_iteration_artifact,
)
from .judge import (
    Judge,
    JudgeContext,
    JudgeResult,
    average_judge_scores,
    create_judge,
    normalize_tool_calls,
)
from .suite import (
    CaseFn,
    CaseResult,
    EvalCase,
    EvalCaseContext,
    EvalRegistry,
    describe_eval,
    get_registry,
    run_case,
)
from .summary import (
    CorrectnessLiftSummary,
    HarnessComparisonDiagnostic,
    HarnessComparisonReport,
    HarnessEvalSetReport,
    HarnessObservation,
    HarnessPairComparison,
    PairedMetricSummary,
    format_harness_comparison_report,
    summarize_harness_comparisons,
)

__all__ = [
    "CorrectnessLiftSummary",
    "EVAL_HARNESS_ITERATION_ARTIFACT",
    "EvalCase",
    "EvalCaseContext",
    "EvalHarnessRow",
    "EvalInput",
    "EvalRegistry",
    "FunctionHarness",
    "Harness",
    "HarnessComparisonDiagnostic",
    "HarnessComparisonReport",
    "HarnessContext",
    "HarnessEvalSetReport",
    "HarnessObservation",
    "HarnessPairComparison",
    "HarnessRun",
    "JsonValue",
    "Judge",
    "JudgeContext",
    "JudgeResult",
    "PairedMetricSummary",
    "PI_EVAL_SOURCES_ARTIFACT",
    "PI_SESSION_SNAPSHOT_ARTIFACT",
    "CaseFn",
    "CaseResult",
    "average_judge_scores",
    "canonicalize_json",
    "create_harness",
    "create_judge",
    "derive_eval_group_key",
    "derive_input_key",
    "describe_eval",
    "eval_harness_table",
    "format_harness_comparison_report",
    "get_registry",
    "normalize_tool_calls",
    "parse_eval_harness_iteration_artifact",
    "persist_eval_artifact_references",
    "run_case",
    "summarize_harness_comparisons",
]
