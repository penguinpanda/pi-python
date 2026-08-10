"""
Unit tests for Provider dispatch logic（注册表模式）。
"""

from unittest.mock import AsyncMock, patch

import pytest

from pi_ai.api.api_provider_registry import (
    ApiProvider,
    register_api_provider,
    reset_api_providers,
)
from pi_ai.utils._event_stream import AssistantMessageEventStream
from pi_ai._types import (
    AssistantMessage,
    Context,
    Model,
)
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.provider import Provider, create_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    model_id: str = "test-model",
    provider: str = "test-provider",
    api: str = "openai-completions",
) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        api=api,
        name=model_id,
        input=["text"],
        output=["text"],
    )


def _make_provider(api_kind: str = "completions", base_url: str | None = None) -> Provider:
    """Create a minimal Provider for testing dispatch."""
    auth = EnvApiKeyAuth(display_name="Test", env_vars=["TEST_API_KEY"])
    return Provider(
        id="test-provider",
        name="Test Provider",
        auth=auth,
        models=[_make_model()],
        _api_kind=api_kind,  # type: ignore[arg-type]
        base_url=base_url,
    )


def _context() -> Context:
    return Context(
        messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ]
    )


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


def _stub_provider(api: str, record: list) -> ApiProvider:
    """同步返回 EventStream 的注册表 stub。"""

    def _stream(model: Model, context: Context, options=None):
        record.append((model, context, options))
        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": _done_message(model)})
        fake_stream.end()
        return fake_stream

    return ApiProvider(api=api, stream=_stream, streamSimple=_stream)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_api_providers()
    yield
    reset_api_providers()


# ---------------------------------------------------------------------------
# Provider.stream() 注册表分发
# ---------------------------------------------------------------------------


class TestProviderStreamDispatch:
    """Provider.stream() 按 model.api 从注册表分发。"""

    @pytest.mark.asyncio
    async def test_completions_dispatch(self):
        """model.api='openai-completions' → 调用注册的 completions 实现。"""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions", base_url="https://api.test.com")
        model = _make_model()

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(model, _context())

        received_model, received_context, received_options = record[0]
        assert received_model is model
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == "https://api.test.com"

    @pytest.mark.asyncio
    async def test_responses_dispatch(self):
        """model.api='openai-responses' → 调用注册的 responses 实现。"""
        record: list = []
        register_api_provider(_stub_provider("openai-responses", record), source_id="test")
        provider = _make_provider(api_kind="responses", base_url="https://api.openai.com/v1")
        model = _make_model(api="openai-responses")

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(model, _context())

        received_options = record[0][2]
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_responses_empty_base_url(self):
        """base_url=None 时注入空字符串。"""
        record: list = []
        register_api_provider(_stub_provider("openai-responses", record), source_id="test")
        provider = _make_provider(api_kind="responses", base_url=None)
        model = _make_model(api="openai-responses")

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(model, _context())

        received_options = record[0][2]
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == ""

    @pytest.mark.asyncio
    async def test_completions_default_empty_base_url(self):
        """completions 无 base_url 时注入空字符串。"""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions", base_url=None)

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream(_make_model(), _context())

        assert record[0][2]["base_url"] == ""

    @pytest.mark.asyncio
    async def test_api_key_option_overrides_default(self):
        """StreamOptions.api_key 覆盖默认解析并跳过 resolve_api_key。"""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions")
        options = {"api_key": "sk-override"}

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(_make_model(), _context(), options)

        mock_resolve.assert_not_called()
        assert record[0][2]["api_key"] == "sk-override"

    @pytest.mark.asyncio
    async def test_no_auth_skips_key_resolution(self):
        """auth=None（本地服务）跳过 resolve_api_key，传占位 key。"""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = Provider(
            id="keyless",
            name="Keyless",
            auth=None,
            models=[_make_model(provider="keyless")],
            _api_kind="completions",
            base_url="http://localhost:11434/v1",
        )

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(_make_model(provider="keyless"), _context())

        mock_resolve.assert_not_called()
        assert record[0][2]["api_key"] == "ollama"
        assert record[0][2]["base_url"] == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_unknown_api_raises(self):
        """注册表无对应 API 协议时抛错。"""
        provider = _make_provider(api_kind="completions")
        model = _make_model(api="no-such-api")
        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with pytest.raises(ValueError, match="No API provider registered"):
                await provider.stream(model, _context())


# ---------------------------------------------------------------------------
# Provider.complete()
# ---------------------------------------------------------------------------


class TestProviderComplete:
    """Provider.complete() — non-streaming convenience method."""

    @pytest.mark.asyncio
    async def test_complete_returns_assistant_message(self):
        """complete() waits for stream.result() and returns AssistantMessage."""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions")
        model = _make_model()

        expected_msg = AssistantMessage(
            role="assistant",
            content=[{"type": "text", "text": "Hello"}],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={"input": 5, "output": 1, "cache_read": 0, "cache_write": 0, "total_tokens": 6},
            stop_reason="stop",
            error_message=None,
            timestamp=0,
        )

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": expected_msg})

        def _stub_stream(model, context, options=None):
            return fake_stream

        register_api_provider(
            ApiProvider(
                api="openai-completions",
                stream=_stub_stream,
                streamSimple=_stub_stream,
            ),
            source_id="test",
        )

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            result = await provider.complete(model, _context())

        assert result == expected_msg
        assert result["role"] == "assistant"
        assert result["content"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_complete_with_options(self):
        """complete() forwards StreamOptions to stream()."""
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions")
        options = {"temperature": 0.5, "max_tokens": 100}

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.complete(_make_model(), _context(), options)

        received_options = record[0][2]
        assert received_options["temperature"] == 0.5
        assert received_options["max_tokens"] == 100
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == ""


# ---------------------------------------------------------------------------
# Provider.stream_simple() / complete_simple()
# ---------------------------------------------------------------------------


class TestProviderStreamSimple:
    """Provider.stream_simple() 按 model.api 从注册表 streamSimple 分发。"""

    @pytest.mark.asyncio
    async def test_stream_simple_dispatches(self):
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions", base_url="https://api.test.com")
        model = _make_model()
        options = {"reasoning": "low"}

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            await provider.stream_simple(model, _context(), options)

        received_model, received_context, received_options = record[0]
        assert received_model is model
        assert received_options["api_key"] == "sk-test"
        assert received_options["base_url"] == "https://api.test.com"
        assert received_options["reasoning"] == "low"

    @pytest.mark.asyncio
    async def test_complete_simple_returns_assistant_message(self):
        record: list = []
        register_api_provider(_stub_provider("openai-completions", record), source_id="test")
        provider = _make_provider(api_kind="completions")
        model = _make_model()

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            result = await provider.complete_simple(model, _context())

        assert result["role"] == "assistant"
        assert result["stop_reason"] == "stop"


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Factory function create_provider()."""

    def test_default_api_kind(self):
        auth = EnvApiKeyAuth(display_name="Test", env_vars=["KEY"])
        p = create_provider("p1", "Provider 1", auth, [])
        assert p.id == "p1"
        assert p.name == "Provider 1"
        assert p._api_kind == "completions"
        assert p.base_url is None

    def test_custom_api_kind_and_base_url(self):
        auth = EnvApiKeyAuth(display_name="Test", env_vars=["KEY"])
        p = create_provider(
            "p2",
            "Provider 2",
            auth,
            [],
            api_kind="responses",
            base_url="https://custom.api.com/v1",
        )
        assert p._api_kind == "responses"
        assert p.base_url == "https://custom.api.com/v1"


# ---------------------------------------------------------------------------
# Provider._stream_fn（自定义流函数）
# ---------------------------------------------------------------------------


class TestProviderStreamFn:
    """设置 _stream_fn 时跳过 API Key 解析与注册表分发。"""

    def test_create_provider_wires_stream_fn(self):
        async def stream_fn(model, context, options):
            return AssistantMessageEventStream()

        p = create_provider(
            "custom",
            "Custom",
            None,
            [_make_model(provider="custom")],
            stream_fn=stream_fn,
        )
        assert p._stream_fn is stream_fn

    @pytest.mark.asyncio
    async def test_stream_fn_bypasses_key_and_dispatch(self):
        """设置 _stream_fn 后：不调用 resolve_api_key，不进入注册表分发。"""

        async def stream_fn(model, context, options):
            stream = AssistantMessageEventStream()
            stream.push(
                {
                    "type": "done",
                    "reason": "stop",
                    "message": AssistantMessage(
                        role="assistant",
                        content=[{"type": "text", "text": "faux"}],
                        api=model.api,
                        provider=model.provider,
                        model=model.id,
                        stop_reason="stop",
                    ),
                }
            )
            stream.end()
            return stream

        provider = _make_provider(api_kind="completions")
        provider._stream_fn = stream_fn

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            result = await provider.complete(_make_model(), _context())

        mock_resolve.assert_not_called()
        assert result["content"] == [{"type": "text", "text": "faux"}]

    @pytest.mark.asyncio
    async def test_stream_fn_receives_model_context_options(self):
        """stream_fn 收到原始参数（不经过 key 解析）。"""
        received = {}

        async def stream_fn(model, context, options):
            received["model"] = model
            received["context"] = context
            received["options"] = options
            stream = AssistantMessageEventStream()
            stream.push(
                {
                    "type": "done",
                    "reason": "stop",
                    "message": AssistantMessage(
                        role="assistant",
                        content=[],
                        api=model.api,
                        provider=model.provider,
                        model=model.id,
                        stop_reason="stop",
                    ),
                }
            )
            stream.end()
            return stream

        provider = _make_provider(api_kind="completions")
        provider._stream_fn = stream_fn
        options = {"api_key": "sk-test"}

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(_make_model(), _context(), options)

        mock_resolve.assert_not_called()
        assert received["model"] == _make_model()
        assert received["context"] == _context()
        assert received["options"] is options
