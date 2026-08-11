"""Provider 认证链路接入 OAuth 的测试。"""

from __future__ import annotations

import pytest

from pi_ai._types import AssistantMessage, Context, Model
from pi_ai.api.api_provider_registry import (
    ApiProvider,
    register_api_provider,
    reset_api_providers,
)
from pi_ai.auth.credential_store import InMemoryCredentialStore
from pi_ai.provider import Provider
from pi_ai.types import now_ms
from pi_ai.utils._event_stream import AssistantMessageEventStream


class _OAuth:
    name = "oauth-test"

    def __init__(self) -> None:
        self.to_auth_calls = 0

    async def login(self, interaction):  # type: ignore[no-untyped-def]
        return {"type": "oauth", "access": "a", "refresh": "r", "expires": now_ms() + 3600_000}

    async def refresh(self, credential, signal=None):  # type: ignore[no-untyped-def]
        return credential

    async def to_auth(self, credential):  # type: ignore[no-untyped-def]
        self.to_auth_calls += 1
        return {
            "api_key": credential["access"],
            "headers": {"X-OAuth": "1"},
        }


class _Auth:
    def __init__(self, oauth: _OAuth) -> None:
        self.oauth = oauth


def _model() -> Model:
    return Model(id="m", provider="oauth-provider", api="oauth-test")


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "Hi"}])  # type: ignore[typeddict-unknown-key]


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
    def _stream(model: Model, context: Context, options=None):  # type: ignore[no-untyped-def]
        record.append(dict(options or {}))
        stream = AssistantMessageEventStream()
        stream.push({"type": "done", "reason": "stop", "message": _done_message(model)})
        stream.end()
        return stream

    return ApiProvider(api=api, stream=_stream, streamSimple=_stream)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_api_providers()
    yield
    reset_api_providers()


@pytest.mark.asyncio
async def test_provider_stream_uses_oauth_credential() -> None:
    record: list[dict] = []
    register_api_provider(_stub_provider("oauth-test", record), source_id="test-oauth")
    oauth = _OAuth()
    store = InMemoryCredentialStore()
    await store.write(
        "oauth-provider",
        {"type": "oauth", "access": "acc", "refresh": "ref", "expires": now_ms() + 3600_000},
    )
    provider = Provider(
        id="oauth-provider",
        name="OAuth Provider",
        auth=_Auth(oauth),  # type: ignore[arg-type]
        models=[_model()],
        base_url="https://oauth.example.com",
    )
    provider._credential_store = store

    await provider.stream(_model(), _context())

    assert oauth.to_auth_calls == 1
    assert record[0]["api_key"] == "acc"
    assert record[0]["base_url"] == "https://oauth.example.com"
    assert record[0]["headers"] == {"X-OAuth": "1"}


@pytest.mark.asyncio
async def test_provider_stream_explicit_key_skips_oauth() -> None:
    record: list[dict] = []
    register_api_provider(_stub_provider("oauth-test", record), source_id="test-oauth")
    oauth = _OAuth()
    provider = Provider(
        id="oauth-provider",
        name="OAuth Provider",
        auth=_Auth(oauth),  # type: ignore[arg-type]
        models=[_model()],
    )

    await provider.stream(_model(), _context(), {"api_key": "sk-explicit"})

    assert oauth.to_auth_calls == 0
    assert record[0]["api_key"] == "sk-explicit"


@pytest.mark.asyncio
async def test_provider_stream_missing_oauth_raises() -> None:
    register_api_provider(_stub_provider("oauth-test", []), source_id="test-oauth")
    provider = Provider(
        id="oauth-provider",
        name="OAuth Provider",
        auth=_Auth(_OAuth()),  # type: ignore[arg-type]
        models=[_model()],
    )

    with pytest.raises(ValueError, match="No auth configured"):
        await provider.stream(_model(), _context())
