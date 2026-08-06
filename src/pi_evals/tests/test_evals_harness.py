"""Pi coding-agent harness 测试（faux provider，零网络）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pi_ai import Models
from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime

from pi_evals.harness import (
    PiCodingAgentHarnessOptions,
    create_pi_coding_agent_harness,
    resolve_model_selection,
    _make_runtime,
)
from pi_evals.vitest_evals.harness import HarnessContext


def _faux_runtime(responses: list | None = None) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses(responses or [faux_assistant_message("Paris")])
    models.add_provider(core.provider)
    return ModelRuntime(models, store)


class TestResolveModelSelection:
    def test_env_vars(self, monkeypatch):
        monkeypatch.setenv("PI_PROVIDER", "openai")
        monkeypatch.setenv("PI_MODEL", "gpt-5")
        assert resolve_model_selection() == {"provider": "openai", "id": "gpt-5"}

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("PI_PROVIDER", "openai")
        monkeypatch.setenv("PI_MODEL", "gpt-5")
        assert resolve_model_selection({"provider": "faux", "id": "faux-1"}) == {
            "provider": "faux",
            "id": "faux-1",
        }

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("PI_PROVIDER", raising=False)
        monkeypatch.delenv("PI_MODEL", raising=False)
        with pytest.raises(ValueError):
            resolve_model_selection()


@pytest.mark.asyncio
async def test_default_runtime_resolves_real_models(monkeypatch, tmp_path):
    """默认运行时与 CLI 一致：能解析真实模型（deepseek/deepseek-v4-flash）。"""
    monkeypatch.setattr("pi_evals.harness.get_agent_dir", lambda: tmp_path)
    runtime = await _make_runtime(PiCodingAgentHarnessOptions())
    model = runtime.get_model("deepseek", "deepseek-v4-flash")
    assert model is not None
    assert model.provider == "deepseek"


@pytest.mark.asyncio
async def test_run_basic_prompt_collects_transcript_usage_and_snapshot():
    runtime = _faux_runtime()
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
    )
    result = await harness.run(
        "What's the capital of France? Respond with only the city name.",
        HarnessContext(),
    )
    assert result.output == "Paris"
    assert result.errors == []
    assert result.usage["provider"] == "faux"
    assert result.usage["model"] == "faux-1"
    assert result.usage["inputTokens"] >= 0
    assert result.usage["outputTokens"] > 0
    assert result.usage["totalTokens"] > 0
    assert result.timings["totalMs"] >= 0
    roles = [event["role"] for event in result.events if event["type"] == "message"]
    assert "user" in roles
    assert "assistant" in roles
    assert isinstance(result.artifacts["runId"], str)
    session_snapshot = result.artifacts["piSessionJsonl"]
    assert isinstance(session_snapshot, str)
    assert "What's the capital of France?" in session_snapshot


@pytest.mark.asyncio
async def test_run_steps_support_reload():
    runtime = _faux_runtime([faux_assistant_message("Paris"), faux_assistant_message("Paris")])
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
    )
    result = await harness.run(
        [{"type": "reload"}, {"type": "prompt", "content": "hi"}],
        HarnessContext(),
    )
    assert result.errors == []
    assert result.output == "Paris"


@pytest.mark.asyncio
async def test_transform_system_prompt_and_output_fn():
    runtime = _faux_runtime()

    def output(args):
        return {
            "response": args["response"],
            "prompt": args["session"]._agent.state.system_prompt,
            "toolCount": len(args["session"]._agent.state.tools),
        }

    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
        transform_system_prompt=lambda prompt: "custom eval system prompt",
        output=output,
    )
    result = await harness.run("hi", HarnessContext())
    assert result.output == {
        "response": "Paris",
        "prompt": "custom eval system prompt",
        "toolCount": 0,
    }
    assert result.errors == []


@pytest.mark.asyncio
async def test_transform_system_prompt_must_not_be_empty():
    runtime = _faux_runtime()
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        transform_system_prompt=lambda _prompt: "   ",
    )
    with pytest.raises(ValueError, match="must not be empty"):
        await harness.run("hi", HarnessContext())


@pytest.mark.asyncio
async def test_early_error_cleans_temp_dir():
    runtime = _faux_runtime()
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        transform_system_prompt=lambda _prompt: "   ",
    )
    before = set(Path(tempfile.gettempdir()).glob("pi-eval-*"))
    with pytest.raises(ValueError, match="must not be empty"):
        await harness.run("hi", HarnessContext())
    after = set(Path(tempfile.gettempdir()).glob("pi-eval-*"))
    assert after == before


@pytest.mark.asyncio
async def test_unexpected_stop_reason_raises():
    runtime = _faux_runtime(
        [faux_assistant_message("oops", stop_reason="error", error_message="boom")]
    )
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
    )
    with pytest.raises(RuntimeError, match="boom"):
        await harness.run("hi", HarnessContext())


HELLO_EXTENSION_SOURCE = """\
from pi_coding_agent.extensions import ToolDefinition

def _hello(tool_call_id, params, signal=None, on_update=None, context=None):
    return {"content": [{"type": "text", "text": f"Hello, {params['name']}!"}]}

def create_extension(api):
    api.register_tool(ToolDefinition(
        name="hello",
        description="Say hello",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        execute=_hello,
    ))
"""


def _write_hello_extension(cwd: Path) -> None:
    extensions_dir = cwd / ".pi" / "extensions"
    extensions_dir.mkdir(parents=True)
    (extensions_dir / "hello.py").write_text(HELLO_EXTENSION_SOURCE, encoding="utf-8")


@pytest.mark.asyncio
async def test_extension_reload_end_to_end():
    runtime = _faux_runtime(
        [
            faux_assistant_message("created"),
            faux_assistant_message([faux_tool_call("hello", {"name": "Bob"})]),
            faux_assistant_message("Hello, Bob!"),
        ]
    )
    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        reload_setup=_write_hello_extension,
    )
    result = await harness.run(
        [
            {"type": "prompt", "content": "Create the hello extension."},
            {"type": "reload"},
            {"type": "prompt", "content": "Use the hello tool to greet Bob."},
        ],
        HarnessContext(),
    )
    assert result.errors == []
    assert result.output == "Hello, Bob!"
    tool_calls = [event for event in result.events if event["type"] == "tool_call"]
    tool_results = [event for event in result.events if event["type"] == "tool_result"]
    assert tool_calls
    assert tool_calls[-1]["name"] == "hello"
    assert tool_results
    assert tool_results[-1]["content"] == "Hello, Bob!"
    assert result.usage["toolCalls"] >= 1
