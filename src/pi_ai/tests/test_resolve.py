"""认证解析（resolve_provider_auth / resolve_stored_oauth）测试。"""

import asyncio

import pytest

from pi_ai.auth import env_api_key_auth
from pi_ai.auth.context import AuthContext
from pi_ai.auth.credential_store import InMemoryCredentialStore
from pi_ai.auth.resolve import (
    ModelsError,
    resolve_provider_auth,
    resolve_stored_oauth,
)
from pi_ai.types import now_ms


class _FakeAuthContext:
    def __init__(self, env: dict[str, str] | None = None):
        self._env = env or {}

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


class _FakeProvider:
    def __init__(self, provider_id="p", auth=None):
        self.id = provider_id
        self.auth = auth


def _oauth_credential(expires: int) -> dict:
    return {"type": "oauth", "access": "a", "refresh": "r", "expires": expires}


class _FakeOAuth:
    name = "Fake OAuth"

    def __init__(self):
        self.refresh_calls = 0

    async def login(self, interaction):
        return _oauth_credential(now_ms() + 3600_000)

    async def refresh(self, credential, signal=None):
        self.refresh_calls += 1
        return _oauth_credential(now_ms() + 3600_000)

    async def to_auth(self, credential):
        return {"api_key": credential["access"]}


@pytest.mark.asyncio
async def test_resolve_stored_oauth_refreshes_once_under_lock():
    store = InMemoryCredentialStore()
    oauth = _FakeOAuth()
    await store.write("p", _oauth_credential(now_ms() - 1000))  # 已过期

    async def resolve():
        return await resolve_stored_oauth(store, "p", oauth, _oauth_credential(now_ms() - 1000))

    results = await asyncio.gather(resolve(), resolve(), resolve())
    # 双重检查锁定：3 个并发请求只刷新 1 次。
    assert oauth.refresh_calls == 1
    assert all(r is not None and r.auth["api_key"] == "a" for r in results)
    stored = await store.read("p")
    assert stored["expires"] > now_ms()  # 刷新结果已持久化


@pytest.mark.asyncio
async def test_resolve_stored_oauth_valid_does_not_refresh():
    store = InMemoryCredentialStore()
    oauth = _FakeOAuth()
    valid = _oauth_credential(now_ms() + 3600_000)
    await store.write("p", valid)
    result = await resolve_stored_oauth(store, "p", oauth, valid)
    assert oauth.refresh_calls == 0
    assert result.auth["api_key"] == "a"


@pytest.mark.asyncio
async def test_resolve_stored_oauth_min_validity_enforced():
    class _ShortLivedOAuth:
        name = "Short"

        async def refresh(self, credential, signal=None):
            return _oauth_credential(now_ms() + 60_000)  # 1 分钟后过期

        async def to_auth(self, credential):
            return {"api_key": credential["access"]}

    store = InMemoryCredentialStore()
    expired = _oauth_credential(now_ms() - 1000)
    await store.write("p", expired)
    with pytest.raises(ModelsError) as excinfo:
        await resolve_stored_oauth(store, "p", _ShortLivedOAuth(), expired, min_oauth_validity_ms=5 * 60_000)
    assert excinfo.value.code == "oauth"


@pytest.mark.asyncio
async def test_resolve_provider_auth_api_key_override():
    provider = _FakeProvider("p", env_api_key_auth("Test", ["TEST_KEY"]))
    store = InMemoryCredentialStore()
    result = await resolve_provider_auth(
        provider, store, _FakeAuthContext(), {"api_key": "sk-explicit"}
    )
    assert result.auth["api_key"] == "sk-explicit"
    assert result.source == "stored credential"


@pytest.mark.asyncio
async def test_resolve_provider_auth_stored_then_env():
    provider = _FakeProvider("p", env_api_key_auth("Test", ["TEST_KEY"]))
    store = InMemoryCredentialStore()
    await store.write("p", {"type": "api_key", "key": "sk-stored"})
    result = await resolve_provider_auth(provider, store, _FakeAuthContext())
    assert result.auth["api_key"] == "sk-stored"

    store2 = InMemoryCredentialStore()
    result = await resolve_provider_auth(
        provider, store2, _FakeAuthContext({"TEST_KEY": "sk-env"})
    )
    assert result.auth["api_key"] == "sk-env"
    assert result.source == "TEST_KEY"


@pytest.mark.asyncio
async def test_resolve_provider_auth_oauth_path():
    oauth = _FakeOAuth()

    class _OAuthProvider:
        id = "p"
        auth = type("Auth", (), {"oauth": oauth})()

    store = InMemoryCredentialStore()
    await store.write("p", _oauth_credential(now_ms() + 3600_000))
    result = await resolve_provider_auth(_OAuthProvider(), store, _FakeAuthContext())
    assert result.source == "OAuth"
    assert result.auth["api_key"] == "a"
