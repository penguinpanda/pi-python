"""eval runner（对齐 TS packages/evals/scripts/run-evals.mjs）。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path

from .vitest_evals.harness_table import (
    EVAL_HARNESS_ITERATION_ARTIFACT,
    parse_eval_harness_iteration_artifact,
)
from .vitest_evals.summary import (
    HarnessObservation,
    format_harness_comparison_report,
    summarize_harness_comparisons,
)
from .vitest_evals.suite import CaseResult, get_registry, run_case


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_eval_paths() -> list[Path]:
    package_dir = _package_dir()
    return [package_dir / "smoke_eval.py", package_dir / "extensions_eval.py"]


def _load_module(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Eval module not found: {resolved}")
    if resolved.parent.resolve() == _package_dir().resolve():
        # 内置 eval 模块使用相对导入，须挂在 pi_evals 包命名空间下。
        module_name = f"pi_evals._eval_suite_{uuid.uuid4().hex[:12]}"
    else:
        module_name = f"pi_evals_suite_{uuid.uuid4().hex[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load eval module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)


def _resolve_cli_model(
    provider: str | None,
    model: str | None,
    environment: MutableMapping[str, str],
) -> dict[str, str] | None:
    """CLI 显式选择 > 环境变量；两者必须成对出现（对齐 TS run-evals.mjs）。"""
    has_cli_selection = provider is not None or model is not None
    resolved_provider: str | None
    resolved_model: str | None
    if has_cli_selection:
        if not provider or not model:
            raise ValueError("CLI model selection requires both --provider and --model.")
        resolved_provider = provider.strip()
        resolved_model = model.strip()
    else:
        resolved_provider = environment.get("PI_PROVIDER", "").strip() or None
        resolved_model = environment.get("PI_MODEL", "").strip() or None
        if (resolved_provider is None) != (resolved_model is None):
            raise ValueError("Default model selection requires both PI_PROVIDER and PI_MODEL.")
    if resolved_provider and resolved_model:
        environment["PI_PROVIDER"] = resolved_provider
        environment["PI_MODEL"] = resolved_model
        return {"provider": resolved_provider, "id": resolved_model}
    return None


def _resolve_artifact_dir(artifact_dir: str | None, environment: Mapping[str, str]) -> Path:
    base = _package_dir()
    env_value = environment.get("PI_EVAL_ARTIFACT_DIR", "").strip()
    if env_value:
        path = base / env_value
    elif artifact_dir:
        path = base / artifact_dir
    else:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")
        path = base / ".eval" / f"{timestamp}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_observations(results: list[CaseResult]) -> list[HarnessObservation]:
    observations: list[HarnessObservation] = []
    for result in results:
        run = result.run
        if run is None:
            continue
        iteration = parse_eval_harness_iteration_artifact(
            run.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT)
        )
        if iteration is None:
            continue
        metadata = run.usage.get("metadata") or {}
        if run.errors:
            observations.append(
                HarnessObservation(
                    eval_set=iteration.eval_set,
                    group_key=iteration.group_key,
                    test_name=result.case.name,
                    file=result.case.file,
                    harness=iteration.harness,
                    baseline=iteration.baseline,
                    candidates=iteration.candidates,
                    repetition=iteration.repetition,
                    outcome="errored",
                    total_tokens=run.usage.get("totalTokens"),
                    total_ms=run.timings.get("totalMs"),
                    estimated_cost_usd=metadata.get("estimatedCostUsd"),
                )
            )
        elif result.avg_score is not None:
            observations.append(
                HarnessObservation(
                    eval_set=iteration.eval_set,
                    group_key=iteration.group_key,
                    test_name=result.case.name,
                    file=result.case.file,
                    harness=iteration.harness,
                    baseline=iteration.baseline,
                    candidates=iteration.candidates,
                    repetition=iteration.repetition,
                    outcome="scored",
                    score=result.avg_score,
                    total_tokens=run.usage.get("totalTokens"),
                    total_ms=run.timings.get("totalMs"),
                    estimated_cost_usd=metadata.get("estimatedCostUsd"),
                )
            )
        elif result.failed:
            observations.append(
                HarnessObservation(
                    eval_set=iteration.eval_set,
                    group_key=iteration.group_key,
                    test_name=result.case.name,
                    file=result.case.file,
                    harness=iteration.harness,
                    baseline=iteration.baseline,
                    candidates=iteration.candidates,
                    repetition=iteration.repetition,
                    outcome="errored",
                    total_tokens=run.usage.get("totalTokens"),
                    total_ms=run.timings.get("totalMs"),
                    estimated_cost_usd=metadata.get("estimatedCostUsd"),
                )
            )
        else:
            observations.append(
                HarnessObservation(
                    eval_set=iteration.eval_set,
                    group_key=iteration.group_key,
                    test_name=result.case.name,
                    file=result.case.file,
                    harness=iteration.harness,
                    baseline=iteration.baseline,
                    candidates=iteration.candidates,
                    repetition=iteration.repetition,
                    outcome="unscored",
                    total_tokens=run.usage.get("totalTokens"),
                    total_ms=run.timings.get("totalMs"),
                    estimated_cost_usd=metadata.get("estimatedCostUsd"),
                )
            )
    return observations


async def _run_cases(
    registry,
    artifact_dir: Path,
) -> tuple[list[CaseResult], str]:
    results = [await run_case(case, artifact_dir) for case in registry.cases]
    observations = _collect_observations(results)
    report = format_harness_comparison_report(summarize_harness_comparisons(observations))
    return results, report


def _format_status(results: list[CaseResult]) -> str:
    lines = [f"{'name':<48} {'status':<8} {'score':<8} {'tokens':<8}"]
    lines.append("-" * 76)
    for result in results:
        status = "ok" if not result.failed else "FAIL"
        score = f"{result.avg_score:.2f}" if result.avg_score is not None else "-"
        tokens = "-"
        if result.run is not None:
            total_tokens = result.run.usage.get("totalTokens")
            if isinstance(total_tokens, int):
                tokens = str(total_tokens)
        lines.append(f"{result.case.name:<48} {status:<8} {score:<8} {tokens:<8}")
    failed = [result for result in results if result.failed]
    if failed:
        lines.append("")
        lines.append("Failed runs:")
        for result in failed:
            lines.append(f"  - {result.case.name}: {result.failure}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pi-evals",
        description="Run pi coding-agent evals (TS packages/evals Python port).",
    )
    parser.add_argument("--provider", help="Provider id (requires --model).")
    parser.add_argument("--model", help="Model id (requires --provider).")
    parser.add_argument(
        "--artifact-dir",
        help="Artifact directory under src/pi_evals (default: .eval/<timestamp>_<uuid>).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Eval module .py paths (default: built-in smoke + extensions evals).",
    )
    args = parser.parse_args(argv)

    try:
        model_selection = _resolve_cli_model(args.provider, args.model, os.environ)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    registry = get_registry()
    registry.clear()
    paths = [Path(path) for path in args.paths] or _default_eval_paths()
    for path in paths:
        try:
            _load_module(path)
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)
            return 2

    artifact_dir = _resolve_artifact_dir(args.artifact_dir, os.environ)
    default_model = (
        f"{model_selection['provider']}/{model_selection['id']}"
        if model_selection is not None
        else "none"
    )
    print(f"[eval] default-model={default_model}")
    print(f"[eval] artifacts={artifact_dir}")

    try:
        results, report = asyncio.run(_run_cases(registry, artifact_dir))
    except KeyboardInterrupt:
        print("Eval run interrupted.", file=sys.stderr)
        return 130

    print(_format_status(results))
    if report:
        print("")
        print(report)
    if not results:
        print("No eval cases were registered.", file=sys.stderr)
        return 1
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
