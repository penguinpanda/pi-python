"""eval runner CLI 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_evals import runner
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
core.set_responses([faux_assistant_message("Paris")])
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
    names = [case.name for case in registry.cases]
    assert "Pi Coding Agent smoke" in names
    assert any(name.startswith("system-prompt-without-docs repetition 1") for name in names)
    assert any(name.startswith("default-system-prompt repetition 1") for name in names)
    registry.clear()
