"""OpenRouter loopback 回调流程测试。"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from pi_ai.auth.oauth import openrouter
from pi_ai.auth.oauth.openrouter import _start_loopback_callback


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

    callback_url, wait_for_callback, stop = await _start_loopback_callback(
        "verifier-x", "state-x", asyncio.Event()
    )
    assert callback_url is not None
    assert "/oauth/callback" in callback_url
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{callback_url}?code=abc123&state=state-x")
        assert response.status_code == 200
        assert b"Signed in to OpenRouter" in response.content

        credential = await wait_for_callback()
        assert credential is not None
        assert credential["access"] == "or-key"
        assert captured == {"code": "abc123", "verifier": "verifier-x"}
    finally:
        await stop()


@pytest.mark.asyncio
async def test_loopback_callback_error_and_state_mismatch(monkeypatch) -> None:
    callback_url, wait_for_callback, stop = await _start_loopback_callback(
        "verifier-x", "state-x", asyncio.Event()
    )
    assert callback_url is not None
    try:
        # state 不匹配 → 400，不结束等待
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{callback_url}?code=zzz&state=WRONG")
        assert response.status_code == 400

        # error 路径 → 400 且结束等待（凭证 None）
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{callback_url}?error=access_denied&state=state-x")
        assert response.status_code == 400
        assert await asyncio.wait_for(wait_for_callback(), timeout=5) is None
    finally:
        await stop()


@pytest.mark.asyncio
async def test_stop_releases_server_without_callback() -> None:
    """无回调时 stop 幂等关闭服务器（资源泄漏回归）。"""
    callback_url, _wait, stop = await _start_loopback_callback(
        "verifier-x", "state-x", asyncio.Event()
    )
    assert callback_url is not None
    await stop()
    await stop()  # 幂等
