"""harness 与 reporter 单元测试。"""

from __future__ import annotations

import pytest
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime

from pi_evals.harness import PiCodingAgentHarness, resolve_model_selection
from pi_evals.reporter import report_results


class TestResolveModelSelection:
    def test_env_vars(self, monkeypatch):
        monkeypatch.setenv("PI_PROVIDER", "openai")
        monkeypatch.setenv("PI_MODEL", "gpt-5")
        assert resolve_model_selection() == {"provider": "openai", "id": "gpt-5"}

    def test_explicit_wins(self):
        assert resolve_model_selection(
            {"provider": "faux", "id": "faux-1"},
            {"PI_PROVIDER": "openai", "PI_MODEL": "gpt-5"},
        ) == {"provider": "faux", "id": "faux-1"}

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("PI_PROVIDER", raising=False)
        monkeypatch.delenv("PI_MODEL", raising=False)
        with pytest.raises(ValueError):
            resolve_model_selection()


def _runtime():
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("Paris")])
    models.add_provider(core.provider)
    return ModelRuntime(models, store)


@pytest.mark.asyncio
async def test_run_collects_transcript_and_usage():
    runtime = _runtime()
    harness = PiCodingAgentHarness(
        runtime=runtime,
        model={"provider": "faux", "id": "faux-1"},
    )
    result = await harness.run("capital of France?")
    assert result.output.strip() == "Paris"
    assert result.errors == []
    roles = [event["role"] for event in result.transcript if event["type"] == "message"]
    assert "user" in roles
    assert "assistant" in roles
    assert result.artifacts["sessionId"]
    assert result.usage["provider"] == "faux"
    assert result.usage["model"] == "faux-1"


@pytest.mark.asyncio
async def test_run_steps_support_reload():
    runtime = _runtime()
    harness = PiCodingAgentHarness(
        runtime=runtime,
        model={"provider": "faux", "id": "faux-1"},
    )
    result = await harness.run(
        [
            {"type": "reload"},
            {"type": "prompt", "content": "hi"},
        ]
    )
    assert result.errors == []
    assert result.output.strip() == "Paris"


def test_report_results():
    from pi_evals.harness import EvalResult

    text = report_results(
        [
            EvalResult(output="ok", usage={"totalTokens": 10}, duration_ms=5),
            EvalResult(output="bad", errors=["boom"], usage={"totalTokens": 0}, duration_ms=2),
        ]
    )
    assert "Eval summary" in text
    assert "Failed runs: [2]" in text
    assert "boom" in text
