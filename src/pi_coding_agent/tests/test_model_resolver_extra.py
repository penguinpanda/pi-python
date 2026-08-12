"""ModelResolver 补充测试。"""

from __future__ import annotations

import pytest
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_provider
from pi_ai.providers.openai import openai_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_resolver import (
    ScopedModel,
    find_exact_model_reference_match,
    find_initial_model,
    is_alias,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope_with_diagnostics,
    restore_model_from_session,
    try_match_model,
)
from pi_coding_agent.model_runtime import ModelRuntime


async def _runtime(*, with_auth: bool = True) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    for provider in [openai_provider(), faux_provider().provider]:
        models.add_provider(provider)
    runtime = ModelRuntime(models, store)
    await runtime.get_available()
    if with_auth:
        await store.write("openai", {"type": "api_key", "key": "sk-x"})
        await store.write("faux", {"type": "api_key", "key": "local"})
    return runtime


def _model(id: str, provider: str = "faux", name: str = "") -> Model:
    return Model(id=id, provider=provider, api="openai-completions", name=name)


def test_is_alias() -> None:
    assert is_alias("gpt-5-latest") is True
    assert is_alias("gpt-5") is True
    assert is_alias("gpt-5-20250101") is False


def test_find_exact_match_boundaries() -> None:
    models = [
        _model("faux-1", "faux"),
        _model("faux-1", "faux2"),
    ]
    assert find_exact_model_reference_match("", models) is None
    assert find_exact_model_reference_match("faux/faux-1", models) is not None
    assert find_exact_model_reference_match("faux-1", models) is None


def test_try_match_model_alias_preference() -> None:
    models = [
        _model("gpt-5-20250101", "openai"),
        _model("gpt-5", "openai"),
        _model("gpt-5-latest", "openai"),
    ]
    result = try_match_model("gpt", models)
    assert result is not None
    assert result.id == "gpt-5-latest"
    assert try_match_model("nope", models) is None


def test_parse_model_pattern_colon_without_match() -> None:
    models = [_model("faux-1")]
    result = parse_model_pattern("faux-1:high", models)
    assert result.model is not None
    assert result.thinking_level == "high"
    result = parse_model_pattern("missing:high", models)
    assert result.model is None


async def test_resolve_cli_model_boundaries() -> None:
    runtime = await _runtime(with_auth=False)
    result = resolve_cli_model(cli_provider=None, cli_model=None, model_runtime=runtime)
    assert result.error is None

    runtime_empty = ModelRuntime(
        Models(credentials=AuthStorage.in_memory()), AuthStorage.in_memory()
    )
    result = resolve_cli_model(
        cli_provider=None,
        cli_model="x",
        model_runtime=runtime_empty,
    )
    assert result.error is not None
    assert "No models available" in result.error

    result = resolve_cli_model(
        cli_provider="openai",
        cli_model="custom-model:high",
        model_runtime=runtime,
    )
    assert result.model is not None
    assert result.model.id == "custom-model"
    assert result.thinking_level == "high"

    result = resolve_cli_model(
        cli_provider=None,
        cli_model="not-a-model",
        model_runtime=runtime,
    )
    assert result.error is not None
    assert result.model is None


async def test_resolve_scope_diagnostics() -> None:
    runtime = await _runtime()
    result = await resolve_model_scope_with_diagnostics(
        ["faux/faux-1:bogus", "missing", "*5*"],
        runtime,
    )
    assert any(d.code == "no-match" for d in result.diagnostics)
    assert any(d.code == "invalid-thinking-level" for d in result.diagnostics)
    assert result.scoped_models


async def test_find_initial_model_paths(monkeypatch) -> None:
    runtime = await _runtime()

    with pytest.raises(ValueError, match="Unknown provider"):
        await find_initial_model(
            cli_provider="nope",
            cli_model="gpt-5",
            scoped_models=[],
            is_continuing=False,
            model_runtime=runtime,
        )

    model = _model("scoped")
    result = await find_initial_model(
        cli_provider=None,
        cli_model=None,
        scoped_models=[ScopedModel(model, "high")],
        is_continuing=False,
        model_runtime=runtime,
    )
    assert result.model is model
    assert result.thinking_level == "high"

    result = await find_initial_model(
        cli_provider=None,
        cli_model=None,
        scoped_models=[],
        is_continuing=False,
        default_provider="openai",
        default_model_id="gpt-5-chat-latest",
        model_runtime=runtime,
    )
    assert result.model is not None
    assert result.model.provider == "openai"

    empty = ModelRuntime(Models(credentials=AuthStorage.in_memory()), AuthStorage.in_memory())
    result = await find_initial_model(
        cli_provider=None,
        cli_model=None,
        scoped_models=[],
        is_continuing=False,
        model_runtime=empty,
    )
    assert result.model is None


async def test_restore_model_from_session(capsys, monkeypatch) -> None:
    runtime = await _runtime()
    restored, message = await restore_model_from_session(
        "faux",
        "faux-1",
        None,
        True,
        runtime,
    )
    assert restored is not None
    assert message is None
    assert "Restored model" in capsys.readouterr().out

    monkeypatch.setattr(runtime, "has_configured_auth", lambda provider: False)
    current = _model("fallback")
    restored, message = await restore_model_from_session(
        "faux",
        "faux-1",
        current,
        True,
        runtime,
    )
    assert restored is current
    assert "no auth configured" in message

    restored, message = await restore_model_from_session(
        "faux",
        "faux-1",
        None,
        False,
        runtime,
    )
    assert restored is not None
    assert message is not None
