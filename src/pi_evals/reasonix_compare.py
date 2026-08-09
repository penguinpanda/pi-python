"""Reasonix 与 pi-python 同场景 eval 对比适配器（核心逻辑）。

仅使用标准库，便于在无第三方依赖环境直接运行；真实请求由
`reasonix run` / `context-maintenance-e2e` 二进制发起，本模块负责
工作区、配置、命令组装、指标解析、统一成本重算与报告渲染。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# DeepSeek V4 官方单价（美元 / 1M token），与 pi 生成目录 deepseek.json 一致。
DEEPSEEK_PRICES: dict[str, float] = {
    "input": 0.14,
    "output": 0.28,
    "cache_read": 0.0028,
    "cache_write": 0.0,
}

# 与 pi_evals.extensions_eval / long_session_cache_eval 逐字一致的提示词。
CREATE_EXTENSION_PROMPT = (
    "Create a Pi extension with a hello tool that takes a name and returns a greeting. "
    "For example, passing Bob should return `Hello, Bob!`."
)
USE_HELLO_PROMPT = (
    "Use the hello tool to greet Bob. Respond with exactly the tool's greeting and nothing else."
)
LONG_SESSION_PROMPTS: list[str] = [
    "Read data/big.txt without a limit and repeat its first line exactly.",
    "Use bash to run `wc -l data/big.txt` and reply with only the number.",
    "Read data/target.txt and reply with only the value after ANSWER=.",
    "Use bash to compute 7*6 and reply with only the number.",
    "Use bash to print the first 3 lines of data/big.txt and reply with only those lines.",
]
BIG_FILE_LINES = 3000

REASONIX_CONFIG = """default_model = "deepseek"

[[providers]]
name = "deepseek"
kind = "openai"
base_url = "https://api.deepseek.com"
models = ["deepseek-v4-flash"]
default = "deepseek-v4-flash"
api_key_env = "DEEPSEEK_API_KEY"
context_window = 1000000
"""


def require_api_key() -> None:
    """reasonix 场景与官方 benchmark 都需要 DEEPSEEK_API_KEY。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set; export it before running "
            "compare_reasonix_eval.py scenarios/benchmark"
        )


def write_reasonix_home(home: Path) -> Path:
    """写入隔离的 REASONIX_HOME/.env（reasonix 只从这里读 provider 凭证）。"""
    home.mkdir(parents=True, exist_ok=True)
    env_path = home / ".env"
    env_path.write_text(f"DEEPSEEK_API_KEY={os.environ['DEEPSEEK_API_KEY']}\n", encoding="utf-8")
    env_path.chmod(0o600)
    return home


def unified_cost_usd(*, miss: int, hit: int, output: int, cache_write: int = 0) -> float:
    """按 DeepSeek V4 官方单价统一重算成本（美元）。"""
    return (
        miss * DEEPSEEK_PRICES["input"]
        + hit * DEEPSEEK_PRICES["cache_read"]
        + output * DEEPSEEK_PRICES["output"]
        + cache_write * DEEPSEEK_PRICES["cache_write"]
    ) / 1_000_000


def _big_line(index: int) -> str:
    return f"line-{index:04d}: " + "x" * 50


def setup_workspace(ws: Path, *, long_session: bool) -> None:
    """写入 reasonix.toml 与场景文件；长会话额外生成大文件。"""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "reasonix.toml").write_text(REASONIX_CONFIG, encoding="utf-8")
    if long_session:
        data = ws / "data"
        data.mkdir(parents=True, exist_ok=True)
        lines = [_big_line(i) for i in range(1, BIG_FILE_LINES + 1)]
        (data / "big.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (data / "target.txt").write_text("ANSWER=42\n", encoding="utf-8")


@dataclass(slots=True)
class RunMetrics:
    """reasonix run --metrics 输出的核心字段。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cost: float = 0.0
    steps: int = 0
    prefix_change_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def unified_cost(self) -> float:
        return unified_cost_usd(
            miss=self.cache_miss_tokens,
            hit=self.cache_hit_tokens,
            output=self.completion_tokens,
        )


def parse_run_metrics(path: str | Path) -> RunMetrics:
    """解析 reasonix RunMetrics JSON。字段缺失时按 0 处理。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"metrics JSON must be an object: {path}")
    reasons = raw.get("prefix_change_reason_counts")
    return RunMetrics(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        cache_hit_tokens=int(raw.get("cache_hit_tokens", 0) or 0),
        cache_miss_tokens=int(raw.get("cache_miss_tokens", 0) or 0),
        cost=float(raw.get("cost", 0.0) or 0.0),
        steps=int(raw.get("steps", 0) or 0),
        prefix_change_reason_counts=dict(reasons) if isinstance(reasons, dict) else {},
    )


@dataclass(slots=True)
class PiRun:
    """pi-evals runs.jsonl 中的一次 run。"""

    name: str
    scenario: str
    variant: str
    status: str
    score: float | None
    elapsed_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost: float | None

    @property
    def unified_cost(self) -> float:
        return unified_cost_usd(
            miss=self.input_tokens,
            hit=self.cache_read_tokens,
            output=self.output_tokens,
            cache_write=self.cache_write_tokens,
        )


def _scenario_from_file(file: str) -> str:
    if "long_session_cache_eval" in file:
        return "long"
    if "extensions_eval" in file:
        return "extension"
    return "other"


def _variant_from_name(name: str) -> str:
    if "cache-first" in name:
        return "cache-first"
    if "system-prompt-without-docs" in name:
        return "no-docs"
    return "default"


def load_pi_runs(path: str | Path) -> list[PiRun]:
    """读取 pi-evals artifact 的 runs.jsonl。"""
    runs: list[PiRun] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        test = raw.get("test") or {}
        usage = raw.get("usage") or {}
        meta = usage.get("metadata") or {}
        runs.append(
            PiRun(
                name=str(test.get("name", "")),
                scenario=_scenario_from_file(str(test.get("file", ""))),
                variant=_variant_from_name(str(test.get("name", ""))),
                status=str(test.get("status", "")),
                score=float(raw.get("score"))
                if isinstance(raw.get("score"), (int, float))
                else None,
                elapsed_ms=int((raw.get("timings") or {}).get("totalMs", 0) or 0),
                input_tokens=int(usage.get("inputTokens", 0) or 0),
                output_tokens=int(usage.get("outputTokens", 0) or 0),
                cache_read_tokens=int(meta.get("cacheReadTokens", 0) or 0),
                cache_write_tokens=int(meta.get("cacheWriteTokens", 0) or 0),
                estimated_cost=float(meta["estimatedCostUsd"])
                if isinstance(meta.get("estimatedCostUsd"), (int, float))
                else None,
            )
        )
    return runs


def reasonix_run_command(
    bin_path: str | Path,
    *,
    ws: str | Path,
    metrics: str | Path,
    prompt: str,
    model: str = "deepseek",
    continue_session: bool = False,
) -> list[str]:
    """组装 reasonix run 命令（每次一个 prompt）。"""
    cmd = [
        str(bin_path),
        "run",
        "--dir",
        str(ws),
        "--model",
        model,
        "--permission-mode",
        "auto",
        "--output-format",
        "json",
        "--metrics",
        str(metrics),
    ]
    if continue_session:
        cmd.append("--continue")
    cmd.append(prompt)
    return cmd


@dataclass(slots=True)
class RunOutcome:
    """一次 reasonix run 的执行结果。"""

    command: list[str]
    stdout: str
    stderr: str
    elapsed_ms: int
    ok: bool
    metrics: RunMetrics | None = None


def execute_reasonix(
    cmd: list[str],
    *,
    cwd: str | Path,
    metrics_path: str | Path | None = None,
    timeout: int = 900,
    env_overrides: dict[str, str] | None = None,
) -> RunOutcome:
    """执行 reasonix 命令；成功且给出 metrics 路径时解析指标。"""
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(os.environ), **(env_overrides or {})},
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        ok = proc.returncode == 0
        outcome = RunOutcome(
            command=list(cmd),
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_ms=elapsed,
            ok=ok,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        outcome = RunOutcome(
            command=list(cmd),
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            elapsed_ms=elapsed,
            ok=False,
        )
    if outcome.ok and metrics_path is not None and Path(metrics_path).exists():
        outcome.metrics = parse_run_metrics(metrics_path)
    return outcome


def _extract_response(stdout: str) -> str:
    """从 --output-format json 输出中尽力提取最终回复文本。"""
    text = stdout.strip()
    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return text
        for key in ("response", "output", "text", "message"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return text


def _workspace_files(ws: Path) -> list[Path]:
    excluded_parts = {"reasonix.toml", "sessions", ".reasonix", "archive"}
    files: list[Path] = []
    for path in ws.rglob("*"):
        if not path.is_file():
            continue
        if excluded_parts.intersection(path.parts):
            continue
        files.append(path)
    return files


@dataclass(slots=True)
class ScenarioResult:
    """一个场景一次 repetition 的结果。"""

    scenario: str
    rep: int
    passed: bool
    failures: list[str]
    outputs: list[str]
    metrics: list[RunMetrics]
    elapsed_ms: int
    commands: list[list[str]]

    @property
    def total_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.metrics)

    @property
    def unified_cost(self) -> float:
        return sum(m.unified_cost for m in self.metrics)

    @property
    def reasonix_cost(self) -> float:
        return sum(m.cost for m in self.metrics)

    @property
    def steps(self) -> int:
        return sum(m.steps for m in self.metrics)


def run_extension_rep(
    bin_path: str | Path,
    ws: Path,
    metrics_dir: Path,
    rep: int,
    reasonix_home: Path | None = None,
) -> ScenarioResult:
    """扩展编写场景：create → use（两段提示，--continue 同会话）。"""
    setup_workspace(ws, long_session=False)
    outcomes: list[RunOutcome] = []
    commands: list[list[str]] = []
    for index, prompt in enumerate((CREATE_EXTENSION_PROMPT, USE_HELLO_PROMPT)):
        metrics = metrics_dir / f"extension-rep{rep}-run{index + 1}.json"
        cmd = reasonix_run_command(
            bin_path,
            ws=ws,
            metrics=metrics,
            prompt=prompt,
            continue_session=index > 0,
        )
        commands.append(cmd)
        outcomes.append(
            execute_reasonix(
                cmd,
                cwd=ws,
                metrics_path=metrics,
                env_overrides=({"REASONIX_HOME": str(reasonix_home)} if reasonix_home else None),
            )
        )
    outputs = [_extract_response(o.stdout) for o in outcomes]
    failures: list[str] = []
    if not outcomes[-1].ok:
        detail = (outcomes[-1].stdout.strip() or outcomes[-1].stderr.strip())[:200]
        failures.append(f"reasonix run failed: {detail}")
    if "Hello, Bob!" not in outputs[-1]:
        failures.append('final response did not contain "Hello, Bob!"')
    if not _workspace_files(ws):
        failures.append("no file was created in the workspace")
    return ScenarioResult(
        scenario="extension",
        rep=rep,
        passed=not failures,
        failures=failures,
        outputs=outputs,
        metrics=[o.metrics for o in outcomes if o.metrics is not None],
        elapsed_ms=sum(o.elapsed_ms for o in outcomes),
        commands=commands,
    )


def run_long_rep(
    bin_path: str | Path,
    ws: Path,
    metrics_dir: Path,
    rep: int,
    reasonix_home: Path | None = None,
) -> ScenarioResult:
    """长会话场景：5 段提示同一会话（--continue），逐段保存 metrics。"""
    setup_workspace(ws, long_session=True)
    outcomes: list[RunOutcome] = []
    commands: list[list[str]] = []
    for index, prompt in enumerate(LONG_SESSION_PROMPTS):
        metrics_path = metrics_dir / f"long-rep{rep}-run{index + 1}.json"
        cmd = reasonix_run_command(
            bin_path,
            ws=ws,
            metrics=metrics_path,
            prompt=prompt,
            continue_session=index > 0,
        )
        commands.append(cmd)
        outcomes.append(
            execute_reasonix(
                cmd,
                cwd=ws,
                metrics_path=metrics_path,
                env_overrides=({"REASONIX_HOME": str(reasonix_home)} if reasonix_home else None),
            )
        )
    outputs = [_extract_response(o.stdout) for o in outcomes]
    failures: list[str] = []
    if any(not o.ok for o in outcomes):
        first = next((o for o in outcomes if not o.ok), None)
        detail = (
            (first.stdout if first else "").strip() or (first.stderr if first else "").strip()
        )[:200]
        failures.append(f"reasonix run failed: {detail}")
    if len(outputs) != len(LONG_SESSION_PROMPTS) or any(not o.strip() for o in outputs):
        failures.append("not all 5 prompts produced a response")
    if not any(token in outputs[-1] for token in ("line-0001", "line-0002", "line-0003")):
        failures.append("final response did not include the first lines of big.txt")
    metrics_list = [o.metrics for o in outcomes if o.metrics is not None]
    if sum(m.steps for m in metrics_list) < 4:
        failures.append("total model steps below 4 (tool-round proxy)")
    return ScenarioResult(
        scenario="long",
        rep=rep,
        passed=not failures,
        failures=failures,
        outputs=outputs,
        metrics=metrics_list,
        elapsed_ms=sum(o.elapsed_ms for o in outcomes),
        commands=commands,
    )


def run_scenarios(
    bin_path: str | Path,
    out_dir: str | Path,
    *,
    reps: int = 3,
) -> list[ScenarioResult]:
    """跑两个场景各 reps 次，并把结果写入 out_dir/results.json。"""
    require_api_key()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reasonix_home = write_reasonix_home(out / "reasonix-home")
    results: list[ScenarioResult] = []
    for rep in range(1, reps + 1):
        results.append(
            run_extension_rep(
                bin_path,
                out / f"extension-rep{rep}",
                metrics_dir,
                rep,
                reasonix_home=reasonix_home,
            )
        )
        results.append(
            run_long_rep(
                bin_path,
                out / f"long-rep{rep}",
                metrics_dir,
                rep,
                reasonix_home=reasonix_home,
            )
        )
    _write_json(
        out / "results.json",
        [
            {
                "scenario": r.scenario,
                "rep": r.rep,
                "passed": r.passed,
                "failures": r.failures,
                "outputs": r.outputs,
                "totalTokens": r.total_tokens,
                "unifiedCostUsd": r.unified_cost,
                "reasonixCostUsd": r.reasonix_cost,
                "elapsedMs": r.elapsed_ms,
                "steps": r.steps,
                "metrics": [
                    {
                        "promptTokens": m.prompt_tokens,
                        "completionTokens": m.completion_tokens,
                        "cacheHitTokens": m.cache_hit_tokens,
                        "cacheMissTokens": m.cache_miss_tokens,
                        "cost": m.cost,
                        "steps": m.steps,
                    }
                    for m in r.metrics
                ],
            }
            for r in results
        ],
    )
    return results


def run_benchmark(
    bin_path: str | Path,
    out_dir: str | Path,
    *,
    trials: int = 3,
    idle_seconds: int = 360,
) -> dict[str, Any]:
    """官方 context-maintenance-e2e：seed → 等待 TTL → resume → comprehension。"""
    require_api_key()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "benchmark-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def run(args: list[str]) -> dict[str, Any]:
        cmd = [str(bin_path), *args]
        proc = subprocess.run(
            cmd,
            cwd=str(out),
            capture_output=True,
            text=True,
            timeout=3600,
            env=dict(os.environ),
        )
        return {
            "command": cmd,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    seed = run(["--dir", str(run_dir), "seed"])
    if seed["ok"] and idle_seconds > 0:
        time.sleep(idle_seconds)
    resume = run(["--dir", str(run_dir), "resume"])
    comprehension = run(["--dir", str(run_dir), "--trials", str(trials), "comprehension"])
    result = {
        "seed": seed,
        "resume": resume,
        "comprehension": comprehension,
        "idleSeconds": idle_seconds,
        "trials": trials,
    }
    _write_json(out / "benchmark.json", result)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pi_summary(pi_runs: list[PiRun]) -> dict[str, dict[str, Any]]:
    """按 (scenario, variant) 聚合 pi 侧指标。"""
    grouped: dict[tuple[str, str], list[PiRun]] = {}
    for run in pi_runs:
        if run.scenario == "other":
            continue
        grouped.setdefault((run.scenario, run.variant), []).append(run)
    summary: dict[str, dict[str, Any]] = {}
    for (scenario, variant), runs in sorted(grouped.items()):
        scored = [r for r in runs if r.score is not None]
        pass_rate = (
            _mean([1.0 if r.score is not None and r.score >= 1.0 else 0.0 for r in scored])
            if scored
            else _mean([1.0 if r.status == "passed" else 0.0 for r in runs])
        )
        summary[f"{scenario}:{variant}"] = {
            "passRate": pass_rate,
            "avgTokens": _mean(
                [float(r.input_tokens + r.output_tokens + r.cache_read_tokens) for r in runs]
            ),
            "avgCost": _mean([r.unified_cost for r in runs]),
            "avgLatencyMs": _mean([float(r.elapsed_ms) for r in runs]),
            "n": len(runs),
        }
    return summary


def render_report(
    pi_runs: list[PiRun],
    scenario_results: list[ScenarioResult],
    benchmark: dict[str, Any] | None,
) -> str:
    """渲染对比报告 Markdown。"""
    pi = _pi_summary(pi_runs)
    lines: list[str] = [
        "# Reasonix 与 pi-python 同场景 eval 对比",
        "",
        "> 生成时间：见文件时间；模型：deepseek-v4-flash；repetitions：3。",
        "",
        "## 口径说明",
        "",
        "- pi 侧成本来自 `runs.jsonl` 原始 usage，按 DeepSeek V4 官方单价重算"
        "（input $0.14/M、output $0.28/M、cache_read $0.0028/M、cache_write $0）。"
        "- pi 侧通过率取 `runs.jsonl` 的 judge score（score=1 通过）；"
        "  旧 artifact 无 score 时退化为完成率（status=passed）。",
        "- Reasonix 侧成本由 `--metrics` 的 hit/miss/completion 按同一单价重算；"
        "其自报 Cost 字段仅作参考。",
        "- 扩展编写提示词与 pi-evals 逐字一致；Reasonix 判分做最小适配"
        "（最终回复含 `Hello, Bob!` 且工作区生成文件），已如实标注。",
        "- 长会话提示词与 pi-evals 一致；工具调用数用 metrics.steps 合计作为代理。",
        "",
        "## pi-python 结果（runs.jsonl）",
        "",
        "| 场景:变体 | 通过率 | 平均 tokens | 平均成本(USD) | 平均时延(ms) | 样本 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, value in pi.items():
        lines.append(
            f"| {key} | {value['passRate'] * 100:.0f}% | {value['avgTokens']:.0f} "
            f"| ${value['avgCost']:.6f} | {value['avgLatencyMs']:.0f} | {value['n']} |"
        )
    lines += ["", "## Reasonix 结果（CLI 适配器）", ""]
    if scenario_results:
        grouped: dict[str, list[ScenarioResult]] = {}
        for result in scenario_results:
            grouped.setdefault(result.scenario, []).append(result)
        lines.append("| 场景 | 通过率 | 平均 tokens | 平均成本(USD) | 平均时延(ms) | 平均 steps |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for scenario, results in sorted(grouped.items()):
            lines.append(
                f"| {scenario} | {_mean([1.0 if r.passed else 0.0 for r in results]) * 100:.0f}% "
                f"| {_mean([float(r.total_tokens) for r in results]):.0f} "
                f"| ${_mean([r.unified_cost for r in results]):.6f} "
                f"| {_mean([float(r.elapsed_ms) for r in results]):.0f} "
                f"| {_mean([float(r.steps) for r in results]):.1f} |"
            )
    else:
        lines.append("（尚未运行：请先执行 `compare_reasonix_eval.py scenarios`）")
    lines += ["", "## Reasonix 官方 benchmark（context-maintenance-e2e）", ""]
    if benchmark:
        for key in ("seed", "resume", "comprehension"):
            item = benchmark.get(key) or {}
            lines.append(f"### {key}")
            lines.append("```")
            lines.append(str(item.get("stdout", "(no output)")))
            lines.append("```")
    else:
        lines.append("（尚未运行：请先执行 `compare_reasonix_eval.py benchmark`）")
    lines += ["", "## 判分适配说明", ""]
    lines.append(
        "- Reasonix 没有 pi 的扩展/reload 语义，扩展场景以"
        "「最终回复包含 Hello, Bob! + 工作区生成文件」作为通过标准。",
    )
    lines.append(
        "- 长会话场景与 pi-evals 判定一致，仅把「工具调用 ≥4」替换为「metrics.steps 合计 ≥4」。",
    )
    lines.append("")
    return "\n".join(lines)
