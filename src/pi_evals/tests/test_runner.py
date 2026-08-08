"""eval runner CLI 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_evals import runner
from pi_evals.vitest_evals.harness import HarnessRun
from pi_evals.vitest_evals.suite import CaseResult, EvalCase
from pi_evals.vitest_evals.suite import get_registry


def test_resolve_cli_model_requires_pair(monkeypatch):
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_MODEL", raising=False)
    with pytest.raises(ValueError, match="requires both"):
        runner._resolve_cli_model("openai", None, {})


def test_resolve_cli_model_env_fallback(monkeypatch):
    env = {"PI_PROVIDER": "openai", "PI_MODEL": "gpt-5"}
    assert runner._resolve_cli_model(None, None, env) == {
        "provider": "openai",
        "id": "gpt-5",
    }
    assert env["PI_PROVIDER"] == "openai"


def test_resolve_cli_model_prefers_cli():
    env = {"PI_PROVIDER": "openai", "PI_MODEL": "gpt-5"}
    assert runner._resolve_cli_model("faux", "faux-1", env) == {
        "provider": "faux",
        "id": "faux-1",
    }


_PASSING_MODULE = """\
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_evals.harness import create_pi_coding_agent_harness
from pi_evals.vitest_evals.suite import describe_eval

store = AuthStorage.in_memory()
models = Models(credentials=store)
core = faux_provider()
core.set_responses([faux_assistant_message("Paris")] * 8)
models.add_provider(core.provider)
runtime = ModelRuntime(models, store)

harness = create_pi_coding_agent_harness(
    model={"provider": "faux", "id": "faux-1"},
    runtime=runtime,
    no_tools=True,
)

@describe_eval("tmp smoke", harness=harness)
async def _case(ctx):
    result = await ctx.run("capital of France?")
    assert result.output == "Paris"
"""


_FAILING_MODULE = _PASSING_MODULE.replace(
    'assert result.output == "Paris"',
    'assert result.output == "Nope"',
)

_TABLE_MODULE = """\
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_evals.harness import create_pi_coding_agent_harness
from pi_evals.vitest_evals.harness_table import eval_harness_table
from pi_evals.vitest_evals.suite import describe_eval

store = AuthStorage.in_memory()
models = Models(credentials=store)
core = faux_provider()
core.set_responses([faux_assistant_message("Paris")] * 8)
models.add_provider(core.provider)
runtime = ModelRuntime(models, store)


def _make(name):
    return create_pi_coding_agent_harness(
        name=name,
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
    )


for _row in eval_harness_table(
    "tmp table",
    baseline=_make("baseline"),
    candidate=_make("candidate"),
):
    @describe_eval(f"{_row.name} repetition {_row.repetition}", harness=_row.harness)
    async def _case(ctx):
        await ctx.run("hi")
"""


def _write_module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "suite.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_main_runs_passing_eval(tmp_path):
    module = _write_module(tmp_path, _PASSING_MODULE)
    artifact_dir = tmp_path / "artifacts"
    code = runner.main(["--artifact-dir", str(artifact_dir), str(module)])
    assert code == 0
    runs_file = artifact_dir / "runs.jsonl"
    assert runs_file.exists()
    record = json.loads(runs_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["test"]["status"] == "passed"
    assert record["usage"]["totalTokens"] > 0


def test_main_returns_nonzero_on_failure(tmp_path):
    module = _write_module(tmp_path, _FAILING_MODULE)
    artifact_dir = tmp_path / "artifacts"
    code = runner.main(["--artifact-dir", str(artifact_dir), str(module)])
    assert code == 1
    record = json.loads((artifact_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["test"]["status"] == "failed"


def test_main_repetitions_controls_harness_table(tmp_path):
    module = _write_module(tmp_path, _TABLE_MODULE)
    artifact_dir = tmp_path / "artifacts"
    code = runner.main(["--repetitions", "2", "--artifact-dir", str(artifact_dir), str(module)])
    assert code == 0
    lines = (artifact_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_main_rejects_non_positive_repetitions(tmp_path, capsys):
    module = _write_module(tmp_path, _TABLE_MODULE)
    code = runner.main(["--repetitions", "0", str(module)])
    assert code == 2
    assert "positive integer" in capsys.readouterr().err


def test_main_rejects_incomplete_cli_model(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_MODEL", raising=False)
    code = runner.main(["--provider", "openai"])
    assert code == 2
    assert "requires both" in capsys.readouterr().err


def test_main_rejects_missing_module(tmp_path, capsys):
    code = runner.main(["--artifact-dir", str(tmp_path / "out"), str(tmp_path / "nope.py")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_builtin_eval_modules_register_cases():
    registry = get_registry()
    registry.clear()
    package_dir = Path(runner.__file__).resolve().parent
    runner._load_module(package_dir / "smoke_eval.py")
    runner._load_module(package_dir / "extensions_eval.py")
    runner._load_module(package_dir / "long_session_cache_eval.py")
    names = [case.name for case in registry.cases]
    assert "Pi Coding Agent smoke" in names
    assert any(name.startswith("system-prompt-without-docs repetition 1") for name in names)
    assert any(name.startswith("default-system-prompt repetition 1") for name in names)
    assert any(name.startswith("default-system-prompt-cache-first repetition 1") for name in names)
    registry.clear()


class _NoopHarness:
    name = "noop"

    async def run(self, input, context):
        return HarnessRun(output="ok")


def test_format_status_prints_judge_notes():
    case = EvalCase(name="case", file="f.py", harness=_NoopHarness(), fn=lambda ctx: None)
    result = CaseResult(
        case=case,
        run=HarnessRun(output="x"),
        failed=False,
        judge_metadata={"extension": {"score": 0.0, "rationale": "final response mismatch"}},
    )
    text = runner._format_status([result])
    assert "Judge notes" in text
    assert "final response mismatch" in text
