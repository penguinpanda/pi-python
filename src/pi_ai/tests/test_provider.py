"""
Unit tests for Provider dispatch logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_ai._event_stream import AssistantMessageEventStream
from pi_ai._types import (
    AssistantMessage,
    Context,
    DeltaEvent,
    DoneEvent,
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


# ---------------------------------------------------------------------------
# Provider.stream() dispatch
# ---------------------------------------------------------------------------

class TestProviderStreamDispatch:
    """Provider.stream() correctly dispatches to Completions or Responses API."""

    @pytest.mark.asyncio
    async def test_completions_dispatch(self):
        """_api_kind='completions' → calls chat_completions_stream."""
        provider = _make_provider(api_kind="completions", base_url="https://api.test.com")
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        # Create a real stream to return from the mock
        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)) as mock_completions:
                with patch("pi_ai.provider.responses_stream") as mock_responses:
                    await provider.stream(model, context)

        # Verify correct API was called
        mock_completions.assert_called_once_with(
            model, context, "sk-test", "https://api.test.com", None
        )
        mock_responses.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_dispatch(self):
        """_api_kind='responses' → calls responses_stream with base_url."""
        provider = _make_provider(api_kind="responses", base_url="https://api.openai.com/v1")
        model = _make_model(api="openai-responses")
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.responses_stream", new=AsyncMock(return_value=fake_stream)) as mock_responses:
                with patch("pi_ai.provider.chat_completions_stream") as mock_completions:
                    await provider.stream(model, context)

        mock_responses.assert_called_once_with(
            model, context, "sk-test", "https://api.openai.com/v1", None
        )
        mock_completions.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_empty_base_url(self):
        """When base_url is None/empty, empty string is passed to responses_stream."""
        provider = _make_provider(api_kind="responses", base_url=None)
        model = _make_model(api="openai-responses")
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.responses_stream", new=AsyncMock(return_value=fake_stream)) as mock_responses:
                await provider.stream(model, context)

        # base_url should be "" when None
        assert mock_responses.call_args[0][2] == "sk-test"  # api_key
        assert mock_responses.call_args[0][3] == ""  # base_url

    @pytest.mark.asyncio
    async def test_completions_default_empty_base_url(self):
        """completions dispatch with no base_url set → passes empty string."""
        provider = _make_provider(api_kind="completions", base_url=None)
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)) as mock_completions:
                await provider.stream(model, context)

        assert mock_completions.call_args[0][3] == ""  # base_url

    @pytest.mark.asyncio
    async def test_api_key_option_overrides_default(self):
        """StreamOptions.apiKey overrides default resolution and skips resolve_api_key."""
        provider = _make_provider(api_kind="completions")
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        options = {"apiKey": "sk-override"}

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)) as mock_completions:
                await provider.stream(model, context, options)

        # resolve_api_key should NOT be called when apiKey is provided
        mock_resolve.assert_not_called()
        assert mock_completions.call_args[0][2] == "sk-override"  # api_key arg

    @pytest.mark.asyncio
    async def test_no_auth_skips_key_resolution(self):
        """auth=None（本地服务）跳过 resolve_api_key，传空 api_key。"""
        provider = Provider(
            id="keyless",
            name="Keyless",
            auth=None,
            models=[_make_model(provider="keyless")],
            _api_kind="completions",
            base_url="http://localhost:11434/v1",
        )
        model = _make_model(provider="keyless")
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)) as mock_completions:
                await provider.stream(model, context)

        mock_resolve.assert_not_called()
        assert mock_completions.call_args[0][2] == "ollama"  # api_key 占位值
        assert mock_completions.call_args[0][3] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Provider.complete()
# ---------------------------------------------------------------------------

class TestProviderComplete:
    """Provider.complete() — non-streaming convenience method."""

    @pytest.mark.asyncio
    async def test_complete_returns_assistant_message(self):
        """complete() waits for stream.result() and returns AssistantMessage."""
        provider = _make_provider(api_kind="completions")
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        expected_msg = AssistantMessage(
            role="assistant",
            content=[{"type": "text", "text": "Hello"}],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={"input": 5, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 6},
            stopReason="stop",
            errorMessage=None,
            timestamp=0,
        )

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": expected_msg})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)):
                result = await provider.complete(model, context)

        assert result == expected_msg
        assert result["role"] == "assistant"
        assert result["content"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_complete_with_options(self):
        """complete() forwards StreamOptions to stream()."""
        provider = _make_provider(api_kind="completions")
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])
        options = {"temperature": 0.5, "maxTokens": 100}

        fake_stream = AssistantMessageEventStream()
        fake_stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
            role="assistant", content=[], api=model.api,
            provider=model.provider, model=model.id,
            usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
            stopReason="stop", errorMessage=None, timestamp=0,
        )})

        with patch("pi_ai.provider.resolve_api_key", new=AsyncMock(return_value="sk-test")):
            with patch("pi_ai.provider.chat_completions_stream", new=AsyncMock(return_value=fake_stream)) as mock_completions:
                await provider.complete(model, context, options)

        mock_completions.assert_called_once_with(
            model, context, "sk-test", "", options  # base_url defaults to ""
        )


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
            "p2", "Provider 2", auth, [],
            api_kind="responses",
            base_url="https://custom.api.com/v1",
        )
        assert p._api_kind == "responses"
        assert p.base_url == "https://custom.api.com/v1"


# ---------------------------------------------------------------------------
# Provider._stream_fn（自定义流函数）
# ---------------------------------------------------------------------------

class TestProviderStreamFn:
    """设置 _stream_fn 时跳过 API Key 解析与 api_kind 分发。"""

    def test_create_provider_wires_stream_fn(self):
        async def stream_fn(model, context, options):
            return AssistantMessageEventStream()

        p = create_provider(
            "custom", "Custom", None, [_make_model(provider="custom")],
            stream_fn=stream_fn,
        )
        assert p._stream_fn is stream_fn

    @pytest.mark.asyncio
    async def test_stream_fn_bypasses_key_and_dispatch(self):
        """设置 _stream_fn 后：不调用 resolve_api_key，不进入 completions/responses。"""

        async def stream_fn(model, context, options):
            stream = AssistantMessageEventStream()
            stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
                role="assistant",
                content=[{"type": "text", "text": "faux"}],
                api=model.api,
                provider=model.provider,
                model=model.id,
                stopReason="stop",
            )})
            stream.end()
            return stream

        provider = _make_provider(api_kind="completions")
        provider._stream_fn = stream_fn
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            with patch("pi_ai.provider.chat_completions_stream") as mock_completions:
                with patch("pi_ai.provider.responses_stream") as mock_responses:
                    result = await provider.complete(model, context)

        mock_resolve.assert_not_called()
        mock_completions.assert_not_called()
        mock_responses.assert_not_called()
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
            stream.push({"type": "done", "reason": "stop", "message": AssistantMessage(
                role="assistant", content=[], api=model.api,
                provider=model.provider, model=model.id,
                stopReason="stop",
            )})
            stream.end()
            return stream

        provider = _make_provider(api_kind="completions")
        provider._stream_fn = stream_fn
        model = _make_model()
        context = Context(messages=[
            {"role": "user", "content": "Hi"}  # type: ignore[typeddict-unknown-key]
        ])
        options = {"apiKey": "sk-test"}

        with patch("pi_ai.provider.resolve_api_key") as mock_resolve:
            await provider.stream(model, context, options)

        mock_resolve.assert_not_called()
        assert received["model"] is model
        assert received["context"] is context
        assert received["options"] is options
