"""api_provider_registry（对齐 TS compat.ts apiProviderRegistry）测试。"""

import pytest

from unittest.mock import AsyncMock, patch

from pi_ai._types import AssistantMessage, Context, Model
from pi_ai.api.api_provider_registry import (
    ApiProvider,
    complete,
    complete_simple,
    get_api_provider,
    get_api_providers,
    invoke_api_stream,
    register_api_provider,
    register_builtin_api_providers,
    reset_api_providers,
    stream,
    stream_simple,
    unregister_api_providers,
)
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.provider import Provider
from pi_ai.utils._event_stream import AssistantMessageEventStream


def _model(api: str = "openai-completions", provider: str = "openai") -> Model:
    return Model(id="m", provider=provider, api=api)


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi"}])


def _done_message(model: Model) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total_tokens": 0},
        stop_reason="stop",
        error_message=None,
        timestamp=0,
    )


def _stub(api: str, record: list) -> ApiProvider:
    """注册一个同步返回 EventStream 的 stub。"""

    def _stream(model: Model, context: Context, options=None):
        record.append((model, context, options))
        event_stream = AssistantMessageEventStream()
        event_stream.push({"type": "done", "reason": "stop", "message": _done_message(model)})
        event_stream.end()
        return event_stream

    return ApiProvider(api=api, stream=_stream, streamSimple=_stream)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_api_providers()
    yield
    reset_api_providers()


# ---------------------------------------------------------------------------
# 注册表基础操作
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_builtins_registered(self):
        for api in ("openai-completions", "openai-responses", "pi-messages"):
            assert get_api_provider(api) is not None, api

    def test_register_and_get(self):
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="test")
        entry = get_api_provider("my-api")
        assert entry is not None
        assert entry.api == "my-api"
        assert get_api_provider("missing") is None

    def test_register_overwrites_same_api(self):
        record: list = []
        register_api_provider(_stub("dup", record), source_id="a")
        register_api_provider(_stub("dup", record), source_id="b")
        providers = [p for p in get_api_providers() if p.api == "dup"]
        assert len(providers) == 1

    def test_unregister_by_source(self):
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="test")
        assert get_api_provider("my-api") is not None
        unregister_api_providers("test")
        assert get_api_provider("my-api") is None
        # 内置条目不受影响。
        assert get_api_provider("openai-completions") is not None

    def test_register_builtins_does_not_clobber(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="custom")
        register_builtin_api_providers()
        # 已注册条目（含自定义）不被内置注册覆盖。
        entry = get_api_provider("openai-completions")
        assert entry is not None

    def test_mismatched_api_raises(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        entry = get_api_provider("openai-completions")
        assert entry is not None
        with pytest.raises(ValueError, match="Mismatched api"):
            entry.stream(_model(api="openai-responses"), _context(), None)

    def test_reset_restores_builtins(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        reset_api_providers()
        assert get_api_provider("openai-completions") is not None


# ---------------------------------------------------------------------------
# Provider.stream 注册表分发
# ---------------------------------------------------------------------------


class TestProviderRegistryDispatch:
    def _make_provider(self, api_kind: str = "completions", base_url: str | None = None) -> Provider:
        return Provider(
            id="test-provider",
            name="Test Provider",
            auth=EnvApiKeyAuth(display_name="Test", env_vars=["TEST_API_KEY"]),
            models=[_model()],
            _api_kind=api_kind,  # type: ignore[arg-type]
            base_url=base_url,
        )

    @pytest.mark.asyncio
    async def test_completions_dispatch_injects_key_and_base_url(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        provider = self._make_provider(api_kind="completions", base_url="https://api.test.com")
        model = _model()

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(model, _context())

        received_model, received_context, received_options = record[0]
        assert received_model is model
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == "https://api.test.com"

    @pytest.mark.asyncio
    async def test_responses_dispatch_by_model_api(self):
        record: list = []
        register_api_provider(_stub("openai-responses", record), source_id="test")
        provider = self._make_provider(api_kind="responses", base_url="https://api.openai.com/v1")
        model = _model(api="openai-responses")

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(model, _context())

        received_options = record[0][2]
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_custom_api_protocol_dispatches(self):
        """新 API 协议注册到注册表后 Provider 即可分发（核心目标）。"""
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="custom")
        provider = self._make_provider(api_kind="completions")
        model = _model(api="my-api")

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            result = await provider.complete(model, _context())

        assert record[0][0] is model
        assert result["model"] == "m"

    @pytest.mark.asyncio
    async def test_api_key_option_overrides_default(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        provider = self._make_provider(api_kind="completions")
        options = {"api_key": "sk-override", "temperature": 0.5}

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(_model(), _context(), options)

        mock_resolve.assert_not_called()
        received_options = record[0][2]
        assert received_options["api_key"] == "sk-override"
        assert received_options["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_no_auth_skips_key_resolution(self):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        provider = Provider(
            id="keyless",
            name="Keyless",
            auth=None,
            models=[_model(provider="keyless")],
            _api_kind="completions",
            base_url="http://localhost:11434/v1",
        )

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(_model(provider="keyless"), _context())

        mock_resolve.assert_not_called()
        received_options = record[0][2]
        assert received_options["api_key"] == "ollama"
        assert received_options["base_url"] == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_unknown_api_raises(self):
        provider = self._make_provider(api_kind="completions")
        model = _model(api="no-such-api")
        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with pytest.raises(ValueError, match="No API provider registered"):
                await provider.stream(model, _context())


# ---------------------------------------------------------------------------
# 顶层分发（对齐 TS compat.stream / streamSimple / complete / completeSimple）
# ---------------------------------------------------------------------------


class TestTopLevelDispatch:
    @pytest.mark.asyncio
    async def test_stream_with_env_api_key_injection(self, monkeypatch):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        event_stream = stream(_model(provider="openai"), _context())
        assert record[0][2]["api_key"] == "sk-env"
        assert record[0][2].get("base_url") is None
        events = [event async for event in event_stream]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_stream_preserves_explicit_api_key(self, monkeypatch):
        record: list = []
        register_api_provider(_stub("openai-completions", record), source_id="test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        stream(_model(provider="openai"), _context(), {"api_key": "sk-explicit"})
        assert record[0][2]["api_key"] == "sk-explicit"

    @pytest.mark.asyncio
    async def test_stream_simple_dispatches(self):
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="test")
        stream_simple(_model(api="my-api"), _context(), {"reasoning": "high"})
        assert record[0][2]["reasoning"] == "high"

    @pytest.mark.asyncio
    async def test_complete_waits_for_result(self):
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="test")
        message = await complete(_model(api="my-api"), _context())
        assert message["model"] == "m"

    @pytest.mark.asyncio
    async def test_complete_simple(self):
        record: list = []
        register_api_provider(_stub("my-api", record), source_id="test")
        message = await complete_simple(_model(api="my-api"), _context())
        assert message["stop_reason"] == "stop"

    def test_unknown_api_top_level_raises(self):
        with pytest.raises(ValueError, match="No API provider registered"):
            stream(_model(api="no-such-api"), _context())


# ---------------------------------------------------------------------------
# invoke_api_stream（async 上下文条目调用）
# ---------------------------------------------------------------------------


class TestInvokeApiStream:
    @pytest.mark.asyncio
    async def test_async_entry(self):
        async def _async_stream(model, context, options=None):
            event_stream = AssistantMessageEventStream()
            event_stream.push({"type": "done", "reason": "stop", "message": _done_message(model)})
            event_stream.end()
            return event_stream

        result = await invoke_api_stream(_async_stream, _model(), _context())
        events = [event async for event in result]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_sync_entry(self):
        record: list = []
        stub = _stub("my-api", record)
        result = await invoke_api_stream(stub.stream, _model(api="my-api"), _context())
        events = [event async for event in result]
        assert events[-1]["type"] == "done"
