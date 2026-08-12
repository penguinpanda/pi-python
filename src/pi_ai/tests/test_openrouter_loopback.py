"""OpenRouter loopback 回调流程测试。"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pi_ai.auth.oauth import openrouter
from pi_ai.auth.oauth.openrouter import (
    _start_loopback_callback,
    _wait_for_callback_credential,
)


class _Interaction:
    def __init__(self):
        self.signal = asyncio.Event()
        self.notified = None
        self.prompt_calls = []

    def notify(self, event):
        self.notified = event

    async def prompt(self, prompt):
        self.prompt_calls.append(prompt)
        await asyncio.sleep(3600)
        raise AssertionError("manual prompt should not be used")


@pytest.mark.asyncio
async def test_loopback_callback_exchanges_credential(monkeypatch) -> None:
    """浏览器回调命中 loopback server 后完成 exchange 并返回凭证。"""
    captured: dict = {}

    async def fake_exchange(code: str, verifier: str) -> dict:
        captured["code"] = code
        captured["verifier"] = verifier
        return {"type": "oauth", "access": "or-key", "refresh": "", "expires": 2**63}

    monkeypatch.setattr(openrouter, "exchange_authorization_code", fake_exchange)

    callback_url = await _start_loopback_callback("verifier-x", asyncio.Event())
    assert callback_url is not None
    assert "/oauth/callback" in callback_url

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{callback_url}?code=abc123&state=st")
    assert response.status_code == 200
    assert b"Signed in to OpenRouter" in response.content

    credential = await _wait_for_callback_credential()
    assert credential is not None
    assert credential["access"] == "or-key"
    assert captured == {"code": "abc123", "verifier": "verifier-x"}


@pytest.mark.asyncio
async def test_loopback_callback_error_path(monkeypatch) -> None:
    callback_url = await _start_loopback_callback("verifier-x", asyncio.Event())
    assert callback_url is not None

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{callback_url}?error=access_denied")
    assert response.status_code == 400

    credential = await _wait_for_callback_credential()
    assert credential is None
