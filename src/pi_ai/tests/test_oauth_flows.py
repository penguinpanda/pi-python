"""OAuth 流程测试（mock HTTP）。"""

import json

import httpx
import pytest

from pi_ai.auth.oauth import github_copilot, openai_codex, openrouter


class _FakeInteraction:
    def __init__(self, prompt_result="", prompts=None):
        self.prompt_result = prompt_result
        self.prompts = prompts or []
        self.events = []
        self.signal = None

    async def prompt(self, prompt):
        self.prompts.append(prompt)
        return self.prompt_result

    def notify(self, event):
        self.events.append(event)


def _mock_client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ---------------- GitHub Copilot ----------------


@pytest.mark.asyncio
async def test_github_copilot_login(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 300,
                },
            )
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gh-token"})
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(
                200,
                json={
                    "token": "copilot:exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;",
                    "expires_at": 9999999999,
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        github_copilot,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    interaction = _FakeInteraction(prompt_result="")
    credential = await github_copilot.github_copilot_oauth.login(interaction)
    assert credential["type"] == "oauth"
    assert credential["access"].startswith("copilot:")
    assert credential["refresh"] == "gh-token"
    assert "proxy-ep" in credential["access"]
    assert any(e["type"] == "device_code" for e in interaction.events)


def test_copilot_base_url_from_token():
    assert (
        github_copilot.get_base_url_from_token(
            "x;proxy-ep=proxy.individual.githubcopilot.com;y"
        )
        == "https://api.individual.githubcopilot.com"
    )
    assert github_copilot.get_github_copilot_base_url() == "https://api.individual.githubcopilot.com"


# ---------------- OpenAI Codex ----------------


@pytest.mark.asyncio
async def test_openai_codex_device_login(monkeypatch):
    polls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/accounts/deviceauth/usercode":
            return httpx.Response(
                200,
                json={"device_auth_id": "d1", "user_code": "ABCD", "interval": 1},
            )
        if path == "/api/accounts/deviceauth/token":
            polls["n"] += 1
            if polls["n"] == 1:
                return httpx.Response(403, json={"error": {"code": "deviceauth_authorization_pending"}})
            return httpx.Response(
                200,
                json={"authorization_code": "auth-code", "code_verifier": "verifier"},
            )
        if path == "/oauth/token":
            return httpx.Response(
                200,
                json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600},
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        openai_codex,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    interaction = _FakeInteraction()
    credential = await openai_codex.openai_codex_oauth.login(interaction)
    assert credential["type"] == "oauth"
    assert credential["access"] == "acc"
    assert credential["refresh"] == "ref"
    assert polls["n"] == 2


def test_codex_account_id_from_jwt():
    import base64
    import json as _json

    payload = _json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "u-123"}})
    encoded = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    token = f"header.{encoded}.sig"
    assert openai_codex.get_account_id(token) == "u-123"


# ---------------- OpenRouter ----------------


@pytest.mark.asyncio
async def test_openrouter_login_manual_paste(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/keys":
            return httpx.Response(200, json={"key": "sk-or-key", "data": {}})
        return httpx.Response(404)

    monkeypatch.setattr(
        openrouter,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    interaction = _FakeInteraction(
        prompt_result="https://openrouter.ai/auth/callback?code=abc123"
    )
    credential = await openrouter.open_router_oauth.login(interaction)
    assert credential["type"] == "oauth"
    assert credential["access"] == "sk-or-key"
    assert credential["refresh"] == ""
    assert credential["expires"] == openrouter.sys.maxsize


def test_openrouter_parse_authorization_input():
    assert (
        openrouter._parse_authorization_input(
            "https://x.dev/cb?code=abc"
        )
        == "abc"
    )
    assert openrouter._parse_authorization_input("code=xyz&state=s") == "xyz"
    assert openrouter._parse_authorization_input("raw-code") == "raw-code"
    assert openrouter._parse_authorization_input("") is None
    assert openrouter._parse_authorization_input(None) is None


def test_builtin_oauth_providers():
    from pi_ai.auth.oauth import builtin_oauth_providers

    providers = builtin_oauth_providers()
    assert [p[0] for p in providers] == [
        "openai-codex",
        "github-copilot",
        "openrouter",
    ]
