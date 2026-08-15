"""ModelResolver 单元测试（--model / --provider / --models 解析）。"""

from __future__ import annotations

from pi_ai import Models
from pi_ai import create_default_models
from pi_ai.providers.faux import FAUX_MODEL, faux_provider
from pi_ai.providers.openai import openai_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_resolver import (
    find_exact_model_reference_match,
    find_initial_model,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope,
)
from pi_coding_agent.model_runtime import ModelRuntime


async def _make_runtime(providers=None) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    # deepseek 模型由 create_default_models() 统一合并生成目录，
    # 单独调用工厂不再携带模型（provider_map 依赖模型列表）。
    default_models = create_default_models()
    deepseek = default_models.get_provider("deepseek")
    for provider in providers or [
        openai_provider(),
        deepseek,
        faux_provider().provider,
    ]:
        models.add_provider(provider)
    runtime = ModelRuntime(models, store)
    await runtime.get_available()
    return runtime


class TestFindExactMatch:
    async def test_canonical_and_bare_id(self):
        runtime = await _make_runtime()
        models = runtime.get_models()
        assert find_exact_model_reference_match("openai/gpt-5-chat-latest", models) is not None
        assert find_exact_model_reference_match("gpt-5-chat-latest", models) is not None

    async def test_ambiguous_bare_id_returns_none(self):
        runtime = await _make_runtime(
            providers=[
                faux_provider(models=[FAUX_MODEL]).provider,
                faux_provider(models=[FAUX_MODEL], provider="faux2").provider,
            ]
        )
        models = runtime.get_models()
        assert find_exact_model_reference_match("faux-1", models) is None


class TestParseModelPattern:
    async def test_thinking_level_suffix(self):
        runtime = await _make_runtime()
        models = runtime.get_models()
        result = parse_model_pattern("gpt-5.6-luna:high", models)
        assert result.model is not None
        assert result.thinking_level == "high"

    async def test_invalid_thinking_level_warns_in_scope_mode(self):
        runtime = await _make_runtime()
        models = runtime.get_models()
        result = parse_model_pattern("gpt-5.6-luna:bogus", models)
        assert result.model is not None
        assert result.thinking_level is None
        assert result.warning is not None

    async def test_strict_mode_rejects_invalid_suffix(self):
        runtime = await _make_runtime()
        models = runtime.get_models()
        result = parse_model_pattern(
            "gpt-5.6-luna:bogus", models, allow_invalid_thinking_level_fallback=False
        )
        assert result.model is None


class TestResolveCliModel:
    async def test_provider_and_model(self):
        runtime = await _make_runtime()
        result = resolve_cli_model(
            cli_provider="openai",
            cli_model="gpt-5-chat-latest",
            model_runtime=runtime,
        )
        assert result.error is None
        assert result.model is not None
        assert result.model.provider == "openai"

    async def test_provider_slash_model(self):
        runtime = await _make_runtime()
        result = resolve_cli_model(
            cli_provider=None,
            cli_model="deepseek/deepseek-chat",
            model_runtime=runtime,
        )
        assert result.model is not None
        assert result.model.provider == "deepseek"

    async def test_unknown_provider_error(self):
        runtime = await _make_runtime()
        result = resolve_cli_model(
            cli_provider="nope",
            cli_model="gpt-5-chat-latest",
            model_runtime=runtime,
        )
        assert result.error is not None
        assert result.model is None

    async def test_fallback_custom_model_id(self):
        runtime = await _make_runtime()
        result = resolve_cli_model(
            cli_provider="openai",
            cli_model="my-custom-model",
            model_runtime=runtime,
        )
        assert result.error is None
        assert result.model is not None
        assert result.model.id == "my-custom-model"
        assert result.warning is not None


class TestResolveModelScope:
    async def test_glob_scope(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        runtime = await _make_runtime()
        scoped = await resolve_model_scope(["*5.6*"], runtime)
        assert len(scoped) >= 1
        assert all("5.6" in entry.model.id for entry in scoped)

    async def test_explicit_list(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        runtime = await _make_runtime()
        scoped = await resolve_model_scope(["deepseek-v4-flash", "faux/faux-1"], runtime)
        ids = [(entry.model.provider, entry.model.id) for entry in scoped]
        assert ("deepseek", "deepseek-v4-flash") in ids
        assert ("faux", "faux-1") in ids

    async def test_no_match_warns(self, capsys):
        runtime = await _make_runtime()
        scoped = await resolve_model_scope(["does-not-exist"], runtime)
        assert scoped == []
        assert "No models match pattern" in capsys.readouterr().err


class TestFindInitialModel:
    async def test_cli_priority(self):
        runtime = await _make_runtime()
        result = await find_initial_model(
            cli_provider="openai",
            cli_model="gpt-5-chat-latest",
            scoped_models=[],
            is_continuing=False,
            model_runtime=runtime,
        )
        assert result.model.provider == "openai"
        assert result.model.id == "gpt-5-chat-latest"

    async def test_falls_back_to_first_available(self, monkeypatch):
        # 不受外部 API Key 环境影响：仅保留免认证的 faux。
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        runtime = await _make_runtime()
        result = await find_initial_model(
            cli_provider=None,
            cli_model=None,
            scoped_models=[],
            is_continuing=False,
            model_runtime=runtime,
        )
        assert result.model is not None
        assert result.model.provider == "faux"  # 唯一免认证 provider
