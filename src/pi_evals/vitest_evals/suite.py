"""describeEval 移植：eval case 注册与执行。"""

from __future__ import annotations

import inspect
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeAlias

from .artifacts import (
    PI_EVAL_SOURCES_ARTIFACT,
    PI_SESSION_SNAPSHOT_ARTIFACT,
    persist_eval_artifact_references,
)
from .harness import EvalInput, Harness, HarnessContext, HarnessRun, JsonValue
from .harness_table import EVAL_HARNESS_ITERATION_ARTIFACT
from .judge import Judge, JudgeContext, average_judge_scores, normalize_tool_calls
from .summary import HarnessObservation

CaseFn: TypeAlias = Callable[["EvalCaseContext"], Awaitable[None] | None]


@dataclass(slots=True)
class EvalCase:
    """一个 eval case：harness + judge + 断言函数。"""

    name: str
    file: str
    harness: Harness
    fn: CaseFn
    judges: list[Judge] = field(default_factory=list)
    judge_threshold: float | None = 1.0


@dataclass(slots=True)
class EvalCaseContext:
    """case 函数收到的运行上下文（对齐 TS describeEval 的 run/artifacts）。"""

    case: EvalCase
    context: HarnessContext = field(default_factory=HarnessContext)
    last_run: HarnessRun | None = None
    error: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    async def run(self, input: EvalInput) -> HarnessRun:
        """运行 harness；异常转换为带 errors 的失败 run。"""
        try:
            result = await self.case.harness.run(input, self.context)
        except Exception as exc:
            self.error = str(exc)
            result = HarnessRun(
                output=None,
                errors=[str(exc)],
                artifacts=dict(self.context.artifacts),
            )
        self.last_run = result
        return result

    def record_source_artifact(
        self,
        name: str,
        content_type: str,
        body: str,
    ) -> None:
        """记录 eval 生成的源文件 artifact（对齐 recordEvalSourceArtifact）。"""
        sources = self.context.artifacts.get(PI_EVAL_SOURCES_ARTIFACT)
        if not isinstance(sources, list):
            sources = []
            self.context.set_artifact(PI_EVAL_SOURCES_ARTIFACT, sources)
        record: dict[str, JsonValue] = {
            "name": name,
            "contentType": content_type,
            "body": body,
            "bodyEncoding": "utf-8",
        }
        sources.append(record)
        if self.last_run is not None:
            self.last_run.artifacts[PI_EVAL_SOURCES_ARTIFACT] = sources


@dataclass(slots=True)
class CaseResult:
    """单个 case 的执行结果。"""

    case: EvalCase
    run: HarnessRun | None
    failed: bool
    failure: str | None = None
    avg_score: float | None = None
    observation: HarnessObservation | None = None


class EvalRegistry:
    """进程内 eval case 注册表（runner 加载模块时填充）。"""

    def __init__(self) -> None:
        self._cases: list[EvalCase] = []

    @property
    def cases(self) -> list[EvalCase]:
        return list(self._cases)

    def clear(self) -> None:
        self._cases = []

    def describe(
        self,
        name: str,
        *,
        harness: Harness,
        judges: list[Judge] | None = None,
        judge_threshold: float | None = 1.0,
    ) -> Callable[[CaseFn], CaseFn]:
        def decorator(fn: CaseFn) -> CaseFn:
            source = inspect.getsourcefile(fn)
            rel_file = Path(source).name if source else ""
            try:
                rel_file = os.path.relpath(source or "", Path.cwd())
            except ValueError:
                pass
            self._cases.append(
                EvalCase(
                    name=name,
                    file=rel_file,
                    harness=harness,
                    fn=fn,
                    judges=list(judges or []),
                    judge_threshold=judge_threshold,
                )
            )
            return fn

        return decorator


_EVAL_REGISTRY = EvalRegistry()


def describe_eval(
    name: str,
    *,
    harness: Harness,
    judges: list[Judge] | None = None,
    judge_threshold: float | None = 1.0,
) -> Callable[[CaseFn], CaseFn]:
    """注册一个 eval case（对齐 TS describeEval 的 it 回调）。"""
    return _EVAL_REGISTRY.describe(
        name,
        harness=harness,
        judges=judges,
        judge_threshold=judge_threshold,
    )


def get_registry() -> EvalRegistry:
    return _EVAL_REGISTRY


def _run_id(ctx: EvalCaseContext) -> str:
    run = ctx.last_run
    if run is not None:
        artifact_run_id = run.artifacts.get("runId")
        if isinstance(artifact_run_id, str) and artifact_run_id:
            return artifact_run_id
    return ctx.run_id


def _persist_run(
    case: EvalCase,
    ctx: EvalCaseContext,
    run: HarnessRun,
    artifact_dir: Path,
    failed: bool,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(ctx)
    references = persist_eval_artifact_references(run.artifacts, run_id, artifact_dir)
    metadata = {
        key: value
        for key, value in run.artifacts.items()
        if key
        not in (
            "runId",
            PI_SESSION_SNAPSHOT_ARTIFACT,
            PI_EVAL_SOURCES_ARTIFACT,
            EVAL_HARNESS_ITERATION_ARTIFACT,
        )
    }
    record: dict[str, Any] = {
        "schemaVersion": 1,
        "runId": run_id,
        "test": {
            "id": case.name,
            "file": case.file,
            "name": case.name,
            "fullName": case.name,
            "status": "failed" if failed else "passed",
        },
        "harness": case.harness.name,
        "usage": run.usage,
        "artifacts": references,
    }
    if run.timings:
        record["timings"] = run.timings
    if run.errors:
        record["errors"] = run.errors
    if metadata:
        record["metadata"] = metadata
    runs_file = artifact_dir / "runs.jsonl"
    with runs_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _build_observation(
    case: EvalCase,
    ctx: EvalCaseContext,
    run: HarnessRun,
    avg_score: float | None,
    failed: bool,
) -> HarnessObservation | None:
    iteration = run.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT)
    if not isinstance(iteration, dict):
        return None
    from .harness_table import parse_eval_harness_iteration_artifact

    parsed = parse_eval_harness_iteration_artifact(iteration)
    if parsed is None:
        return None
    metadata = run.usage.get("metadata") or {}
    outcome: str
    if run.errors:
        outcome = "errored"
    elif avg_score is not None:
        outcome = "scored"
    elif failed:
        outcome = "errored"
    else:
        outcome = "unscored"
    return HarnessObservation(
        eval_set=parsed.eval_set,
        group_key=parsed.group_key,
        test_name=case.name,
        file=case.file,
        harness=parsed.harness,
        baseline=parsed.baseline,
        candidates=parsed.candidates,
        repetition=parsed.repetition,
        outcome=outcome,  # type: ignore[arg-type]
        score=avg_score,
        total_tokens=run.usage.get("totalTokens"),
        total_ms=run.timings.get("totalMs"),
        estimated_cost_usd=metadata.get("estimatedCostUsd"),
    )


async def run_case(case: EvalCase, artifact_dir: Path) -> CaseResult:
    """执行单个 case：断言 + judge 评分 + artifact 持久化。"""
    ctx = EvalCaseContext(case=case)
    failure: str | None = None
    try:
        result = case.fn(ctx)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        failure = str(exc)

    run = ctx.last_run
    failed = failure is not None
    if run is not None:
        if run.errors:
            failed = True
            failure = failure or "; ".join(run.errors)
        judge_context = JudgeContext(
            output=run.output,
            events=run.events,
            tool_calls=normalize_tool_calls(run.events),
        )
        avg_score = average_judge_scores(case.judges, judge_context)
        if (
            case.judges
            and case.judge_threshold is not None
            and avg_score is not None
            and avg_score < case.judge_threshold
        ):
            failed = True
            failure = (
                failure or f"judge score {avg_score:.2f} below threshold {case.judge_threshold:.2f}"
            )
    else:
        failed = True
        failure = failure or "eval case did not run the harness"

    if run is not None:
        _persist_run(case, ctx, run, artifact_dir, failed)
    observation = _build_observation(case, ctx, run, avg_score, failed) if run is not None else None
    return CaseResult(
        case=case,
        run=run,
        failed=failed,
        failure=failure,
        avg_score=avg_score,
        observation=observation,
    )


__all__ = [
    "CaseFn",
    "CaseResult",
    "EvalCase",
    "EvalCaseContext",
    "EvalRegistry",
    "describe_eval",
    "get_registry",
    "run_case",
]
