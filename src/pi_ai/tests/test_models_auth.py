"""Models 认证查询/登录 API 测试。"""

from __future__ import annotations

import pytest

from pi_ai import Models
from pi_ai.auth import EnvApiKeyAuth
from pi_ai.auth.context import AuthContext
from pi_ai.provider import Provider
from pi_ai.types import Model, now_ms


class _EnvContext(AuthContext):
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env = env or {}

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


class _FakeOAuth:
    name = "fake"

    def __init__(self) -> None:
        self.login_calls = 0

    async def login(self, interaction):  # type: ignore[no-untyped-def]
        self.login_calls += 1
        return {"type": "oauth", "access": "acc", "refresh": "ref", "expires": now_ms() + 3600_000}

    async def refresh(self, credential, signal=None):  # type: ignore[no-untyped-def]
        return credential

    async def to_auth(self, credential):  # type: ignore[no-untyped-def]
        return {"api_key": credential["access"]}


class _OAuthAuth:
    def __init__(self, oauth: _FakeOAuth) -> None:
        self.oauth = oauth


def _model(provider_id: str) -> Model:
    return Model(id="m", provider=provider_id, api="openai-completions")


def _provider(
    provider_id: str,
    auth: EnvApiKeyAuth | _OAuthAuth | None = None,
) -> Provider:
    return Provider(
        id=provider_id,
        name=provider_id,
        auth=auth,  # type: ignore[arg-type]
        models=[_model(provider_id)],
    )


@pytest.mark.asyncio
async def test_get_auth_uses_auth_context_env() -> None:
    models = Models(auth_context=_EnvContext({"TEST_KEY": "sk-env"}))
    models.add_provider(_provider("p", EnvApiKeyAuth("Test", ["TEST_KEY"])))

    result = await models.get_auth("p")

    assert result is not None
    assert result.auth["api_key"] == "sk-env"
    assert result.source == "TEST_KEY"


@pytest.mark.asyncio
async def test_check_auth_oauth_stored() -> None:
    oauth = _FakeOAuth()
    models = Models()
    models.add_provider(_provider("p", _OAuthAuth(oauth)))
    await models._credentials.write(
        "p",
        {"type": "oauth", "access": "a", "refresh": "r", "expires": now_ms() + 3600_000},
    )

    check = await models.check_auth("p")

    assert check == {"type": "oauth", "source": "OAuth"}


@pytest.mark.asyncio
async def test_login_and_logout_persist_credential() -> None:
    oauth = _FakeOAuth()
    models = Models()
    models.add_provider(_provider("p", _OAuthAuth(oauth)))

    credential = await models.login("p", None)  # type: ignore[arg-type]

    assert credential["type"] == "oauth"
    assert oauth.login_calls == 1
    assert (await models._credentials.read("p")) is not None
    await models.logout("p")
    assert await models._credentials.read("p") is None


@pytest.mark.asyncio
async def test_get_available_filters_unconfigured_providers() -> None:
    models = Models(auth_context=_EnvContext({"TEST_KEY": "sk-env"}))
    models.add_provider(_provider("configured", EnvApiKeyAuth("Test", ["TEST_KEY"])))
    models.add_provider(_provider("missing", EnvApiKeyAuth("Test", ["OTHER_KEY"])))

    available = await models.get_available()

    assert [model.provider for model in available] == ["configured"]
