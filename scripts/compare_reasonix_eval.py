#!/usr/bin/env python3
"""Reasonix 与 pi-python 同场景 eval 对比 CLI。

用法：
    python scripts/compare_reasonix_eval.py scenarios --out <dir> [--reps 3]
    python scripts/compare_reasonix_eval.py benchmark --out <dir> [--trials 3] [--idle-seconds 360]
    python scripts/compare_reasonix_eval.py report --out <dir> --pi-runs <runs.jsonl> [--report <md>]

真实请求需要 DEEPSEEK_API_KEY 环境变量；scenarios/benchmark 会调用
Reasonix 二进制（默认 ~/.local/bin/reasonix 与 context-maintenance-e2e）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_evals.reasonix_compare import (  # noqa: E402
    RunMetrics,
    ScenarioResult,
    load_pi_runs,
    render_report,
    run_benchmark,
    run_scenarios,
)

DEFAULT_REASONIX_BIN = Path.home() / ".local/bin/reasonix"
DEFAULT_BENCH_BIN = Path.home() / ".local/bin/context-maintenance-e2e"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1]
    / "docs/nd_upload/deepseek-prefix-cache/reasonix-comparison.md"
)


def load_results(path: str | Path) -> list[ScenarioResult]:
    """从 run_scenarios 写出的 results.json 还原结果列表。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    results: list[ScenarioResult] = []
    for item in raw:
        metrics = [
            RunMetrics(
                prompt_tokens=int(m.get("promptTokens", 0) or 0),
                completion_tokens=int(m.get("completionTokens", 0) or 0),
                cache_hit_tokens=int(m.get("cacheHitTokens", 0) or 0),
                cache_miss_tokens=int(m.get("cacheMissTokens", 0) or 0),
                cost=float(m.get("cost", 0.0) or 0.0),
                steps=int(m.get("steps", 0) or 0),
            )
            for m in item.get("metrics", [])
        ]
        results.append(
            ScenarioResult(
                scenario=str(item.get("scenario", "")),
                rep=int(item.get("rep", 0)),
                passed=bool(item.get("passed", False)),
                failures=list(item.get("failures", [])),
                outputs=list(item.get("outputs", [])),
                metrics=metrics,
                elapsed_ms=int(item.get("elapsedMs", 0) or 0),
                commands=[],
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scenarios = sub.add_parser("scenarios", help="跑扩展 + 长会话两个场景各 N 次")
    scenarios.add_argument("--out", required=True, help="结果输出目录")
    scenarios.add_argument("--reps", type=int, default=3)
    scenarios.add_argument("--bin", default=str(DEFAULT_REASONIX_BIN))

    benchmark = sub.add_parser("benchmark", help="跑官方 context-maintenance-e2e")
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--trials", type=int, default=3)
    benchmark.add_argument("--idle-seconds", type=int, default=360)
    benchmark.add_argument("--bin", default=str(DEFAULT_BENCH_BIN))

    report = sub.add_parser("report", help="生成对比报告 Markdown")
    report.add_argument("--out", required=True, help="reasonix 结果目录")
    report.add_argument("--pi-runs", required=True, help="pi-evals runs.jsonl")
    report.add_argument("--report", default=str(DEFAULT_REPORT))
    report.add_argument(
        "--benchmark",
        default=None,
        help="benchmark.json 路径（缺省读 --out/benchmark.json）",
    )

    args = parser.parse_args(argv)
    if args.command == "scenarios":
        results = run_scenarios(args.bin, args.out, reps=args.reps)
        passed = sum(1 for r in results if r.passed)
        print(
            f"scenarios done: {passed}/{len(results)} passed -> {Path(args.out) / 'results.json'}"
        )
        return 0
    if args.command == "benchmark":
        result = run_benchmark(
            args.bin,
            args.out,
            trials=args.trials,
            idle_seconds=args.idle_seconds,
        )
        print("benchmark done ->", Path(args.out) / "benchmark.json")
        for key in ("seed", "resume", "comprehension"):
            item = result.get(key) or {}
            print(item.get("stdout", ""))
        return 0
    if args.command == "report":
        out = Path(args.out)
        results: list[ScenarioResult] = []
        results_path = out / "results.json"
        if results_path.exists():
            results = load_results(results_path)
        benchmark: dict | None = None
        bench_path = Path(args.benchmark) if args.benchmark else out / "benchmark.json"
        if bench_path.exists():
            benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
        markdown = render_report(load_pi_runs(args.pi_runs), results, benchmark)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")
        print("report written ->", report_path)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
