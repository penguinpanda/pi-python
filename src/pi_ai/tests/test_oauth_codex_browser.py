"""OpenAI Codex browser login 流程测试。"""

from __future__ import annotations

import asyncio
import urllib.parse

from pi_ai.auth.oauth.openai_codex import (
    BROWSER_REDIRECT_URI,
    _browser_login,
    _extract_code_from_pasted,
    create_authorization_flow,
)


def test_create_authorization_flow_url() -> None:
    flow = create_authorization_flow()
    parsed = urllib.parse.urlparse(flow["url"])
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/oauth/authorize"
    assert query["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
    assert query["redirect_uri"] == [BROWSER_REDIRECT_URI]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"][0]
    assert query["state"] == [flow["state"]]
    assert flow["verifier"]


def test_extract_code_from_pasted() -> None:
    assert _extract_code_from_pasted(f"{BROWSER_REDIRECT_URI}?code=abc123&state=x") == "abc123"
    assert _extract_code_from_pasted("plain-code") is None


class _Interaction:
    def __init__(self):
        self.signal = None
        self.notified = None
        self.prompt_calls = []

    def notify(self, event):
        self.notified = event

    async def prompt(self, prompt):
        self.prompt_calls.append(prompt)
        return f"{BROWSER_REDIRECT_URI}?code=pasted-code&state=x"


async def _fake_token_exchange(code: str, verifier: str, redirect_uri: str):
    assert code == "pasted-code"
    return {
        "access_token": "acc",
        "refresh_token": "ref",
        "expires_in": 3600,
    }


def test_browser_login_manual_paste_fallback(monkeypatch) -> None:
    """回调服务器无响应时走手动粘贴回退并交换 token。"""
    import pi_ai.auth.oauth.openai_codex as codex

    monkeypatch.setattr(codex, "_wait_for_browser_code", _no_server)
    monkeypatch.setattr(codex, "webbrowser", _FakeBrowser)
    monkeypatch.setattr(codex, "exchange_authorization_code", _fake_token_exchange)

    interaction = _Interaction()
    credential = asyncio.run(_browser_login(interaction))
    assert credential["type"] == "oauth"
    assert credential["access"] == "acc"
    assert credential["refresh"] == "ref"
    assert interaction.notified["type"] == "auth_url"
    assert interaction.prompt_calls


async def _no_server(state, signal):
    return None


class _FakeBrowser:
    opened: list = []

    @staticmethod
    def open(url):
        _FakeBrowser.opened.append(url)
