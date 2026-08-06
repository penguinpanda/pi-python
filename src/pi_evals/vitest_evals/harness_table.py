"""evalHarnessTable 移植：baseline/candidate 对比编排。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .harness import (
    EvalInput,
    Harness,
    HarnessContext,
    HarnessRun,
    JsonValue,
    canonicalize_json,
    create_harness,
)

EVAL_HARNESS_ITERATION_ARTIFACT = "vitestEvalsHarnessIteration"


@dataclass(slots=True)
class EvalHarnessRow:
    """evalHarnessTable 的一行：某个 harness 的某次重复。"""

    harness: Harness
    name: str
    repetition: int


@dataclass(slots=True)
class EvalHarnessIterationArtifact:
    """一次对比迭代的元数据（对齐 TS EvalHarnessIterationArtifact）。"""

    schema_version: int = 1
    eval_set: str = ""
    group_key: str = ""
    harness: str = ""
    baseline: str = ""
    candidates: list[str] = field(default_factory=list)
    repetition: int = 0

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schemaVersion": self.schema_version,
            "evalSet": self.eval_set,
            "groupKey": self.group_key,
            "harness": self.harness,
            "baseline": self.baseline,
            "candidates": list(self.candidates),
            "repetition": self.repetition,
        }


def derive_input_key(input: EvalInput) -> str:
    """输入分组键：优先使用输入对象的非空 id，否则用稳定 JSON 的 SHA-256。"""
    if isinstance(input, dict):
        input_id = input.get("id")
        if isinstance(input_id, str) and input_id.strip():
            return input_id.strip()
    canonical = canonicalize_json(input)
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def derive_eval_group_key(input: EvalInput, repetition: int) -> str:
    """对齐 TS deriveEvalGroupKey：输入键 + 重复次数。"""
    return json.dumps([derive_input_key(input), repetition], ensure_ascii=False)


def parse_eval_harness_iteration_artifact(
    value: JsonValue | None,
) -> EvalHarnessIterationArtifact | None:
    """校验并解析迭代 artifact；格式不合法时返回 None。"""
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schemaVersion")
    eval_set = value.get("evalSet")
    group_key = value.get("groupKey")
    harness = value.get("harness")
    baseline = value.get("baseline")
    candidates = value.get("candidates")
    repetition = value.get("repetition")
    typed_candidates: list[str] = []
    if not isinstance(candidates, list):
        return None
    for name in candidates:
        if not isinstance(name, str):
            return None
        typed_candidates.append(name)
    if (
        not isinstance(schema_version, int)
        or schema_version != 1
        or not isinstance(eval_set, str)
        or not isinstance(group_key, str)
        or not isinstance(harness, str)
        or not isinstance(baseline, str)
        or not isinstance(repetition, int)
    ):
        return None
    return EvalHarnessIterationArtifact(
        schema_version=schema_version,
        eval_set=eval_set,
        group_key=group_key,
        harness=harness,
        baseline=baseline,
        candidates=typed_candidates,
        repetition=repetition,
    )


def _wrap_with_iteration_artifact(
    harness: Harness,
    plan: EvalHarnessIterationArtifact,
) -> Harness:
    """包装 harness：写入迭代 artifact，并把错误转换为带 errors 的 run。"""

    async def run(input: EvalInput, context: HarnessContext) -> HarnessRun:
        group_key = derive_eval_group_key(input, plan.repetition)
        artifact = EvalHarnessIterationArtifact(
            schema_version=plan.schema_version,
            eval_set=plan.eval_set,
            group_key=group_key,
            harness=plan.harness,
            baseline=plan.baseline,
            candidates=list(plan.candidates),
            repetition=plan.repetition,
        )
        artifact_value = artifact.to_dict()
        context.set_artifact(EVAL_HARNESS_ITERATION_ARTIFACT, artifact_value)
        try:
            result = await harness.run(input, context)
        except Exception as exc:
            result = HarnessRun(output=None, errors=[str(exc)], artifacts=dict(context.artifacts))
        result.artifacts = {
            **context.artifacts,
            **result.artifacts,
            EVAL_HARNESS_ITERATION_ARTIFACT: artifact_value,
        }
        return result

    return create_harness(harness.name, run)


def eval_harness_table(
    eval_set: str,
    *,
    baseline: Harness,
    candidate: Harness | None = None,
    candidates: list[Harness] | None = None,
    repetitions: int = 1,
) -> list[EvalHarnessRow]:
    """生成 baseline + candidate(s) × repetitions 的 harness 行（对齐 TS evalHarnessTable）。"""
    if not eval_set.strip():
        raise TypeError("evalSet must not be empty.")
    candidate_list = [candidate] if candidate is not None else list(candidates or [])
    if not candidate_list:
        raise TypeError("At least one candidate harness is required.")
    harnesses = [baseline, *candidate_list]
    names = {harness.name for harness in harnesses}
    if len(names) != len(harnesses):
        raise TypeError("Harness names must be unique within an eval set.")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise TypeError("repetitions must be a positive integer.")

    rows: list[EvalHarnessRow] = []
    for repetition in range(1, repetitions + 1):
        for harness in harnesses:
            plan = EvalHarnessIterationArtifact(
                schema_version=1,
                eval_set=eval_set,
                harness=harness.name,
                baseline=baseline.name,
                candidates=[candidate.name for candidate in candidate_list],
                repetition=repetition,
            )
            rows.append(
                EvalHarnessRow(
                    harness=_wrap_with_iteration_artifact(harness, plan),
                    name=harness.name,
                    repetition=repetition,
                )
            )
    return rows


__all__ = [
    "EVAL_HARNESS_ITERATION_ARTIFACT",
    "EvalHarnessIterationArtifact",
    "EvalHarnessRow",
    "derive_eval_group_key",
    "derive_input_key",
    "eval_harness_table",
    "parse_eval_harness_iteration_artifact",
]
