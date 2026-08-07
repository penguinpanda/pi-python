"""vitest-evals 等价物（harness / judge / table / summary / suite）测试。"""

from __future__ import annotations

import json

import pytest

from pi_evals.vitest_evals import (
    HarnessContext,
    HarnessObservation,
    HarnessRun,
    JudgeContext,
    create_harness,
    create_judge,
    derive_eval_group_key,
    eval_harness_table,
    format_harness_comparison_report,
    normalize_tool_calls,
    parse_eval_harness_iteration_artifact,
    run_case,
    summarize_harness_comparisons,
)
from pi_evals.vitest_evals.artifacts import PI_SESSION_SNAPSHOT_ARTIFACT
from pi_evals.vitest_evals.harness import canonicalize_json
from pi_evals.vitest_evals.harness_table import EVAL_HARNESS_ITERATION_ARTIFACT
from pi_evals.vitest_evals.suite import EvalRegistry


class _FakeHarness:
    def __init__(
        self,
        name: str = "fake",
        output: object = "ok",
        errors: list[str] | None = None,
    ) -> None:
        self.name = name
        self._output = output
        self._errors = errors or []

    async def run(self, input, context):
        context.set_artifact("runId", "run-1")
        context.set_artifact(PI_SESSION_SNAPSHOT_ARTIFACT, '{"type":"session"}\n')
        return HarnessRun(
            output=self._output,
            events=[],
            errors=list(self._errors),
            usage={"totalTokens": 5, "metadata": {"estimatedCostUsd": 0.1}},
            timings={"totalMs": 10},
            artifacts=dict(context.artifacts),
        )


class TestCanonicalizeJson:
    def test_sorts_keys(self):
        assert canonicalize_json({"b": 1, "a": [2, {"d": 3, "c": 4}]}) == {
            "a": [2, {"c": 4, "d": 3}],
            "b": 1,
        }

    def test_rejects_non_finite_numbers(self):
        with pytest.raises(TypeError, match="finite"):
            canonicalize_json(float("nan"))

    def test_rejects_circular_references(self):
        value: list = []
        value.append(value)
        with pytest.raises(TypeError, match="circular"):
            canonicalize_json(value)

    def test_rejects_non_json_types(self):
        with pytest.raises(TypeError, match="JSON-serializable"):
            canonicalize_json(object())


def test_create_harness():
    with pytest.raises(ValueError, match="empty"):
        create_harness("  ", lambda input, context: None)

    async def run(input, context):
        context.set_artifact("name", input)
        return HarnessRun(output=input)

    harness = create_harness("demo", run)
    context = HarnessContext()
    assert harness.name == "demo"
    assert context.artifacts == {}


def test_normalize_tool_calls():
    events = [
        {"type": "tool_call", "id": "call-1", "name": "read", "arguments": {"path": "a"}},
        {
            "type": "tool_result",
            "toolCallId": "call-1",
            "name": "read",
            "content": "file content",
            "isError": False,
        },
        {"type": "tool_call", "id": "call-2", "name": "bash", "arguments": {}},
        {
            "type": "tool_result",
            "toolCallId": "call-2",
            "name": "bash",
            "content": "boom",
            "isError": True,
        },
    ]
    calls = normalize_tool_calls(events)
    assert calls[0]["status"] == "ok"
    assert calls[0]["result"] == "file content"
    assert calls[1]["status"] == "error"
    assert calls[1]["error"] == "boom"


def test_judge_coerces_results():
    judge = create_judge(
        "demo",
        lambda ctx: {"score": 1 if ctx.output == "ok" else 0, "metadata": {"reason": "x"}},
    )
    result = judge.evaluate(JudgeContext(output="ok", events=[], tool_calls=[]))
    assert result.score == 1
    assert result.metadata == {"reason": "x"}


def test_extension_judge_accepts_dotted_canonical_import():
    """from pi_coding_agent.extensions import ... 应视为 canonical 包导入。"""
    from pi_evals.extensions_eval import _extension_authoring_judge

    ctx = JudgeContext(
        output={
            "response": "Hello, Bob!",
            "systemPromptHasGuidelines": True,
            "systemPromptHasPiDocs": True,
            "extensionErrors": [],
            "loadedExtensions": [{"path": "hello.py", "tools": ["hello"]}],
            "extensionSource": (
                "from pi_coding_agent.extensions import ToolDefinition\n"
                "def create_extension(api):\n"
                "    pass\n"
            ),
        },
        events=[],
        tool_calls=[
            {
                "name": "hello",
                "status": "ok",
                "arguments": {"name": "Bob"},
                "result": "Hello, Bob!",
            }
        ],
    )
    result = _extension_authoring_judge(ctx)
    assert result["score"] == 1


def test_extension_judge_accepts_extension_without_canonical_import():
    """Python 扩展可直接用 api 对象，不 import pi_coding_agent 也应通过。"""
    from pi_evals.extensions_eval import _extension_authoring_judge

    ctx = JudgeContext(
        output={
            "response": "Hello, Bob!",
            "systemPromptHasGuidelines": True,
            "systemPromptHasPiDocs": True,
            "extensionErrors": [],
            "loadedExtensions": [{"path": "hello.py", "tools": ["hello"]}],
            "extensionSource": ("def create_extension(api):\n    api.register_tool(...)\n"),
        },
        events=[],
        tool_calls=[
            {
                "name": "hello",
                "status": "ok",
                "arguments": {"name": "Bob"},
                "result": "Hello, Bob!",
            }
        ],
    )
    result = _extension_authoring_judge(ctx)
    assert result["score"] == 1


def test_extension_judge_rejects_legacy_imports():
    """pi_evals / pi_tests / tests. 前缀导入仍应拒绝。"""
    from pi_evals.extensions_eval import _extension_authoring_judge

    ctx = JudgeContext(
        output={
            "response": "Hello, Bob!",
            "systemPromptHasGuidelines": True,
            "systemPromptHasPiDocs": True,
            "extensionErrors": [],
            "loadedExtensions": [{"path": "hello.py", "tools": ["hello"]}],
            "extensionSource": (
                "from pi_evals.harness import create_pi_coding_agent_harness\n"
                "def create_extension(api):\n"
                "    pass\n"
            ),
        },
        events=[],
        tool_calls=[
            {
                "name": "hello",
                "status": "ok",
                "arguments": {"name": "Bob"},
                "result": "Hello, Bob!",
            }
        ],
    )
    result = _extension_authoring_judge(ctx)
    assert result["score"] == 0
    assert "eval-only package" in result["metadata"]["rationale"]


def test_extension_authoring_output_falls_back_to_loaded_hello_extension(tmp_path):
    from types import SimpleNamespace

    from pi_evals.extensions_eval import _extension_authoring_output

    ext_dir = tmp_path / "hello_extension"
    ext_dir.mkdir()
    source = "from pi_coding_agent.extensions import ToolDefinition\n"
    (ext_dir / "pi_extension.py").write_text(source, encoding="utf-8")
    extension = SimpleNamespace(path=str(ext_dir / "pi_extension.py"), tools={"hello": object()})
    runner = SimpleNamespace(extensions=[extension])
    session = SimpleNamespace(
        cwd=str(tmp_path),
        extension_runner=runner,
        _agent=SimpleNamespace(state=SimpleNamespace(system_prompt="x")),
    )
    output = _extension_authoring_output("Hello, Bob!", session)
    assert output["extensionSource"] == source
    assert output["loadedExtensions"] == [
        {"path": str(ext_dir / "pi_extension.py"), "tools": ["hello"]}
    ]


def test_extension_authoring_output_prefers_canonical_hello_py(tmp_path):
    from types import SimpleNamespace

    from pi_evals.extensions_eval import _extension_authoring_output

    canonical = tmp_path / ".pi" / "extensions"
    canonical.mkdir(parents=True)
    canonical_source = "import pi_coding_agent\n"
    (canonical / "hello.py").write_text(canonical_source, encoding="utf-8")
    session = SimpleNamespace(
        cwd=str(tmp_path),
        extension_runner=SimpleNamespace(extensions=[]),
        _agent=SimpleNamespace(state=SimpleNamespace(system_prompt="x")),
    )
    output = _extension_authoring_output("Hello, Bob!", session)
    assert output["extensionSource"] == canonical_source


def test_eval_harness_table_validation():
    baseline = _FakeHarness("baseline")
    candidate = _FakeHarness("candidate")
    with pytest.raises(TypeError, match="evalSet"):
        eval_harness_table("  ", baseline=baseline, candidate=candidate)
    with pytest.raises(TypeError, match="candidate"):
        eval_harness_table("s", baseline=baseline)
    with pytest.raises(TypeError, match="unique"):
        eval_harness_table("s", baseline=baseline, candidate=_FakeHarness("baseline"))
    with pytest.raises(TypeError, match="positive"):
        eval_harness_table("s", baseline=baseline, candidate=candidate, repetitions=0)


def test_eval_harness_table_default_repetitions_from_env(monkeypatch):
    monkeypatch.setenv("PI_EVAL_REPETITIONS", "2")
    rows = eval_harness_table(
        "s",
        baseline=_FakeHarness("baseline"),
        candidate=_FakeHarness("candidate"),
    )
    assert len(rows) == 4


def test_eval_harness_table_rejects_invalid_env_repetitions(monkeypatch):
    monkeypatch.setenv("PI_EVAL_REPETITIONS", "abc")
    with pytest.raises(TypeError, match="positive integer"):
        eval_harness_table(
            "s",
            baseline=_FakeHarness("baseline"),
            candidate=_FakeHarness("candidate"),
        )


def test_eval_harness_table_rows_and_group_key():
    baseline = _FakeHarness("baseline")
    candidate = _FakeHarness("candidate")
    rows = eval_harness_table(
        "eval set",
        baseline=baseline,
        candidate=candidate,
        repetitions=2,
    )
    assert [(row.name, row.repetition) for row in rows] == [
        ("baseline", 1),
        ("candidate", 1),
        ("baseline", 2),
        ("candidate", 2),
    ]
    assert derive_eval_group_key("hello", 1) == derive_eval_group_key("hello", 1)
    assert derive_eval_group_key("hello", 1) != derive_eval_group_key("hello", 2)
    assert derive_eval_group_key({"id": "case-a"}, 1) == json.dumps(["case-a", 1])


@pytest.mark.asyncio
async def test_iteration_artifact_written_and_errors_converted():
    baseline = _FakeHarness("baseline")
    candidate = _FakeHarness("candidate", errors=["boom"])
    rows = eval_harness_table("eval set", baseline=baseline, candidate=candidate)
    result = await rows[0].harness.run("input", HarnessContext())
    iteration = parse_eval_harness_iteration_artifact(
        result.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT)
    )
    assert iteration is not None
    assert iteration.eval_set == "eval set"
    assert iteration.baseline == "baseline"
    assert iteration.candidates == ["candidate"]
    failed = await rows[1].harness.run("input", HarnessContext())
    assert failed.errors == ["boom"]
    assert (
        parse_eval_harness_iteration_artifact(failed.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT))
        is not None
    )


def _observation(
    harness: str,
    score: float | None,
    *,
    outcome: str = "scored",
    tokens: int | None = None,
    ms: float | None = None,
    cost: float | None = None,
) -> HarnessObservation:
    return HarnessObservation(
        eval_set="s",
        group_key="g1",
        test_name="t",
        file="f.py",
        harness=harness,
        baseline="baseline",
        candidates=["candidate"],
        repetition=1,
        outcome=outcome,  # type: ignore[arg-type]
        score=score,
        total_tokens=tokens,
        total_ms=ms,
        estimated_cost_usd=cost,
    )


def test_summary_computes_lift_and_deltas():
    report = summarize_harness_comparisons(
        [
            _observation("baseline", 1.0, tokens=100, ms=100.0, cost=1.0),
            _observation("candidate", 0.0, tokens=120, ms=150.0, cost=1.2),
        ]
    )
    comparison = report.eval_sets[0].comparisons[0]
    assert comparison.correctness.lift == -1.0
    assert comparison.correctness.eligible_pairs == 1
    assert comparison.total_tokens.mean_delta == 20
    assert comparison.total_ms.mean_delta == 50
    assert comparison.estimated_cost_usd.mean_delta == 0.2
    formatted = format_harness_comparison_report(report)
    assert "Eval Comparisons" in formatted
    assert "Pass rate" in formatted
    assert "Tokens" in formatted


def test_summary_reports_incomplete_observations():
    report = summarize_harness_comparisons(
        [
            _observation("baseline", 1.0),
            _observation("candidate", None, outcome="errored"),
        ]
    )
    assert report.diagnostics
    assert report.diagnostics[0].reason == "harness-error"
    assert "Incomplete observations" in format_harness_comparison_report(report)


@pytest.mark.asyncio
async def test_run_case_persists_artifacts(tmp_path):
    registry = EvalRegistry()
    harness = _FakeHarness()

    async def case_fn(ctx):
        await ctx.run("hi")

    registry.describe("passing case", harness=harness)(case_fn)
    result = await run_case(registry.cases[0], tmp_path)
    assert not result.failed
    runs_file = tmp_path / "runs.jsonl"
    assert runs_file.exists()
    record = json.loads(runs_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["schemaVersion"] == 1
    assert record["runId"] == "run-1"
    assert record["harness"] == "fake"
    assert record["test"]["status"] == "passed"
    session_refs = [ref for ref in record["artifacts"] if ref["name"] == "session.jsonl"]
    assert session_refs
    session_path = tmp_path / session_refs[0]["path"]
    assert session_path.exists()
    assert session_path.read_text(encoding="utf-8") == '{"type":"session"}\n'


@pytest.mark.asyncio
async def test_run_case_failures(tmp_path):
    registry = EvalRegistry()

    async def failing_case(ctx):
        result = await ctx.run("hi")
        assert result.output == "nope"

    registry.describe(
        "assertion failure",
        harness=_FakeHarness(output="ok"),
    )(failing_case)
    result = await run_case(registry.cases[0], tmp_path)
    assert result.failed
    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["test"]["status"] == "failed"


@pytest.mark.asyncio
async def test_run_case_judge_threshold(tmp_path):
    registry = EvalRegistry()
    judge = create_judge("bad", lambda ctx: {"score": 0.0})

    async def case_fn(ctx):
        await ctx.run("hi")

    registry.describe(
        "scored case",
        harness=_FakeHarness(output="bad"),
        judges=[judge],
        judge_threshold=1.0,
    )(case_fn)
    result = await run_case(registry.cases[0], tmp_path)
    assert result.failed
    assert result.avg_score == 0.0
    assert "below threshold" in (result.failure or "")


@pytest.mark.asyncio
async def test_run_case_persists_judge_score_and_metadata(tmp_path):
    registry = EvalRegistry()
    judge = create_judge(
        "extension",
        lambda ctx: {"score": 0.0, "metadata": {"rationale": "final response mismatch"}},
    )

    async def case_fn(ctx):
        await ctx.run("hi")

    registry.describe(
        "scored case",
        harness=_FakeHarness(output="bad"),
        judges=[judge],
        judge_threshold=None,
    )(case_fn)
    result = await run_case(registry.cases[0], tmp_path)
    assert result.avg_score == 0.0
    assert result.judge_metadata["extension"]["rationale"] == "final response mismatch"
    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["score"] == 0.0
    assert record["judgeMetadata"]["extension"]["rationale"] == "final response mismatch"


@pytest.mark.asyncio
async def test_run_case_records_source_artifact(tmp_path):
    registry = EvalRegistry()

    async def case_fn(ctx):
        await ctx.run("hi")
        ctx.record_source_artifact("hello.py", "text/x-python", "print('hi')")

    registry.describe("source case", harness=_FakeHarness())(case_fn)
    await run_case(registry.cases[0], tmp_path)
    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    source_refs = [ref for ref in record["artifacts"] if ref["name"] == "hello.py"]
    assert source_refs
    source_path = tmp_path / source_refs[0]["path"]
    assert source_path.read_text(encoding="utf-8") == "print('hi')"
