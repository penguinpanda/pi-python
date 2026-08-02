"""CLI OAuth 子命令测试。"""

import pytest

from pi_coding_agent._cli import (
    _auth_list,
    _auth_login,
    _auth_logout,
    _auth_store,
    _run_auth_command,
)


class _FakeFlow:
    name = "Fake OAuth"

    async def login(self, interaction):
        return {
            "type": "oauth",
            "access": "sk-fake",
            "refresh": "rf-fake",
            "expires": 9999999999999,
        }


def _fake_providers():
    return [("fake", "Fake OAuth", _FakeFlow())]


@pytest.mark.asyncio
async def test_auth_login_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pi_coding_agent._cli.builtin_oauth_providers", _fake_providers
    )
    monkeypatch.setattr(
        "pi_coding_agent._cli._auth_store",
        lambda: _auth_store_with(tmp_path),
    )
    code = await _auth_login("fake")
    assert code == 0
    store = _auth_store_with(tmp_path)
    cred = await store.read("fake")
    assert cred is not None
    assert cred["access"] == "sk-fake"


def _auth_store_with(tmp_path):
    from pi_ai.auth.credential_store import FileCredentialStore

    return FileCredentialStore(tmp_path / "auth.json")


@pytest.mark.asyncio
async def test_auth_login_unknown_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pi_coding_agent._cli.builtin_oauth_providers", _fake_providers
    )
    assert await _auth_login("nope") == 1


@pytest.mark.asyncio
async def test_auth_logout(monkeypatch, tmp_path):
    store = _auth_store_with(tmp_path)
    await store.write("fake", {"type": "oauth", "access": "a", "refresh": "r", "expires": 1})
    monkeypatch.setattr(
        "pi_coding_agent._cli._auth_store", lambda: store
    )
    assert await _auth_logout("fake") == 0
    assert await store.read("fake") is None
    assert await _auth_logout(None) == 1


@pytest.mark.asyncio
async def test_auth_list_shows_status(monkeypatch, tmp_path, capsys):
    store = _auth_store_with(tmp_path)
    await store.write("fake", {"type": "oauth", "access": "a", "refresh": "r", "expires": 1})
    monkeypatch.setattr(
        "pi_coding_agent._cli.builtin_oauth_providers", _fake_providers
    )
    monkeypatch.setattr("pi_coding_agent._cli._auth_store", lambda: store)
    assert await _auth_list() == 0
    output = capsys.readouterr().out
    assert "fake" in output
    assert "logged in" in output


@pytest.mark.asyncio
async def test_run_auth_command_dispatch(monkeypatch, tmp_path):
    store = _auth_store_with(tmp_path)
    monkeypatch.setattr("pi_coding_agent._cli._auth_store", lambda: store)
    assert await _run_auth_command(["logout", "fake"]) == 0
    assert await _run_auth_command(["unknown"]) == 1
