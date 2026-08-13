"""OAuth login 主流程测试（mock 网络与回调）。"""

from __future__ import annotations

import asyncio

import pytest

from pi_ai.auth.oauth import openrouter, radius, xai


class _Interaction:
    def __init__(self, prompt_value: str | None = None):
        self.signal = asyncio.Event()
        self.notified: list = []
        self.prompts: list = []
        self._prompt_value = prompt_value

    def notify(self, event):
        self.notified.append(event)

    async def prompt(self, prompt):
        self.prompts.append(prompt)
        if self._prompt_value is None:
            raise RuntimeError("no prompt value")
        return self._prompt_value


# ---------------------------------------------------------------------------
# xAI device-code login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_xai_login_device_code_flow(monkeypatch) -> None:
    async def fake_post_form(url, fields, signal=None):
        if url == xai.XAI_DEVICE_CODE_URL:
            return _FakeResponse(
                200,
                {
                    "device_code": "dc-1",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://x.ai/device",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        if url == xai.XAI_TOKEN_URL and fields.get("grant_type") == (
            "urn:ietf:params:oauth:grant-type:device_code"
        ):
            return _FakeResponse(
                200,
                {
                    "access_token": "xai-access",
                    "refresh_token": "xai-refresh",
                    "expires_in": 3600,
                },
            )
        raise AssertionError(f"unexpected {url}")

    monkeypatch.setattr(xai, "_post_form", fake_post_form)
    interaction = _Interaction()
    credential = await xai.xai_oauth.login(interaction)
    assert credential["access"] == "xai-access"
    assert credential["refresh"] == "xai-refresh"
    assert interaction.notified
    assert interaction.notified[0]["type"] == "device_code"
    assert interaction.notified[0]["user_code"] == "ABCD-EFGH"


@pytest.mark.asyncio
async def test_xai_refresh_keeps_previous_token(monkeypatch) -> None:
    async def fake_post_form(url, fields, signal=None):
        # 刷新响应不返回新 refresh_token → 沿用旧值
        return _FakeResponse(200, {"access_token": "new-access", "expires_in": 3600})

    monkeypatch.setattr(xai, "_post_form", fake_post_form)
    refreshed = await xai.xai_oauth.refresh(
        {"type": "oauth", "access": "old", "refresh": "old-refresh", "expires": 0}
    )
    assert refreshed["access"] == "new-access"
    assert refreshed["refresh"] == "old-refresh"


# ---------------------------------------------------------------------------
# OpenRouter login（手动粘贴路径）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_login_manual_paste(monkeypatch) -> None:
    async def fake_start_loopback(verifier, state, signal):
        return None, None, openrouter._noop  # server 不可用 → FALLBACK

    async def fake_exchange(code, verifier):
        assert code == "or-code"
        return {"type": "oauth", "access": "or-key", "refresh": "", "expires": 2**63}

    monkeypatch.setattr(openrouter, "_start_loopback_callback", fake_start_loopback)
    monkeypatch.setattr(openrouter, "exchange_authorization_code", fake_exchange)
    monkeypatch.setattr(openrouter, "webbrowser", _FakeBrowser)

    interaction = _Interaction(prompt_value="or-code")
    credential = await openrouter.open_router_oauth.login(interaction)
    assert credential["access"] == "or-key"
    assert interaction.notified[0]["type"] == "auth_url"
    assert interaction.prompts


class _FakeBrowser:
    @staticmethod
    def open(url):
        return True


# ---------------------------------------------------------------------------
# Radius login（浏览器回调路径）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_radius_login_browser_callback(monkeypatch) -> None:
    async def fake_discover(gateway):
        return f"{gateway}/oauth/authorize"

    async def fake_wait_for_code(state, signal):
        return "radius-code"

    async def fake_token(gateway, fields, signal):
        assert fields["grant_type"] == "authorization_code"
        assert fields["code"] == "radius-code"
        return {
            "type": "oauth",
            "access": "radius-access",
            "refresh": "radius-refresh",
            "expires": 2**40,
        }

    monkeypatch.setattr(radius, "_discover_authorization_endpoint", fake_discover)
    monkeypatch.setattr(radius, "_wait_for_browser_code", fake_wait_for_code)
    monkeypatch.setattr(radius, "_request_oauth_token", fake_token)
    monkeypatch.setattr(radius, "webbrowser", _FakeBrowser)

    class _GatewayInteraction(_Interaction):
        gateway = "https://radius.example.com"

    interaction = _GatewayInteraction(prompt_value="unused")
    oauth = radius.create_radius_oauth("https://radius.example.com")
    credential = await oauth.login(interaction)
    assert credential["access"] == "radius-access"
    assert interaction.notified[0]["type"] == "auth_url"
    assert "radius.example.com" in interaction.notified[0]["url"]


class _FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body
        self.is_success = 200 <= status < 300

    def json(self):
        return self._body


@pytest.mark.asyncio
async def test_radius_token_error_detail(monkeypatch) -> None:
    """token 请求失败：错误详情拼接进异常。"""

    class _ErrorResponse:
        status_code = 400
        is_success = False
        text = "oops"

        def json(self):
            return {"error": "invalid_grant", "error_description": "code expired"}

    async def fake_post(url, **kwargs):
        return _ErrorResponse()

    def fake_client(*a, **kw):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            post = staticmethod(fake_post)

        return _Client()

    monkeypatch.setattr(radius, "_AsyncClient", fake_client)
    with pytest.raises(RuntimeError, match="code expired"):
        await radius._request_oauth_token(
            "https://gw.example.com",
            {"grant_type": "authorization_code"},
            None,
        )


@pytest.mark.asyncio
async def test_radius_discover_fallback(monkeypatch) -> None:
    """well-known 请求失败时回退默认授权端点。"""
    import httpx

    class _FailClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(radius, "_AsyncClient", lambda *a, **kw: _FailClient())
    endpoint = await radius._discover_authorization_endpoint("https://gw.example.com")
    assert endpoint == "https://gw.example.com/oauth/authorize"


@pytest.mark.asyncio
async def test_radius_token_missing_fields(monkeypatch) -> None:
    """token 响应缺字段报错。"""

    class _BadResponse:
        status_code = 200
        is_success = True

        def json(self):
            return {"access_token": "a"}  # 缺 refresh_token

    async def fake_post(url, **kwargs):
        return _BadResponse()

    def fake_client(*a, **kw):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            post = staticmethod(fake_post)

        return _Client()

    monkeypatch.setattr(radius, "_AsyncClient", fake_client)
    with pytest.raises(RuntimeError, match="missing fields"):
        await radius._request_oauth_token(
            "https://gw.example.com",
            {"grant_type": "refresh_token"},
            None,
        )


def test_openrouter_parse_input_variants() -> None:
    """URL / code= 串 / 原始码三种输入形式。"""
    assert (
        openrouter._parse_authorization_input(f"{openrouter.FALLBACK_CALLBACK_URL}?code=u1&state=x")
        == "u1"
    )
    assert openrouter._parse_authorization_input("code=u2") == "u2"
    assert openrouter._parse_authorization_input("raw-code") == "raw-code"
    assert openrouter._parse_authorization_input(None) is None
    assert openrouter._parse_authorization_input("") is None


@pytest.mark.asyncio
async def test_openrouter_to_auth_and_refresh() -> None:
    credential = {
        "type": "oauth",
        "access": "or-key",
        "refresh": "",
        "expires": 2**63,
    }
    auth = await openrouter.open_router_oauth.to_auth(credential)
    assert auth == {"api_key": "or-key"}
    # refresh 返回原凭证（key 永久）
    assert await openrouter.open_router_oauth.refresh(credential) is credential


def test_xai_response_validation() -> None:
    """_required_string / _positive_number / verification_uri 校验错误路径。"""
    with pytest.raises(RuntimeError, match="device_code"):
        xai._required_string({}, "device_code")
    with pytest.raises(RuntimeError, match="expires_in"):
        xai._positive_number({"expires_in": -1}, "expires_in")
    with pytest.raises(RuntimeError, match="Untrusted verification URI"):
        xai._validate_verification_uri("http://evil.example.com/device")
    assert xai._validate_verification_uri("https://x.ai/device") == "https://x.ai/device"


def test_xai_credentials_from_token_without_expiry() -> None:
    credential = xai._credentials_from_token({"access_token": "a", "refresh_token": "r"})
    assert credential["access"] == "a"
    assert credential["refresh"] == "r"
    # 默认 3600s 生命周期
    assert credential["expires"] > 0


@pytest.mark.asyncio
async def test_xai_poll_error_states(monkeypatch) -> None:
    """authorization_pending → slow_down → access_denied 状态映射。"""
    responses = [
        _FakeResponse(400, {"error": "authorization_pending"}),
        _FakeResponse(400, {"error": "slow_down", "interval": 1}),
        _FakeResponse(400, {"error": "access_denied"}),
    ]

    async def fake_post_form(url, fields, signal=None):
        return responses.pop(0)

    monkeypatch.setattr(xai, "_post_form", fake_post_form)
    device = {
        "device_code": "dc",
        "expires_in_seconds": 300,
        "interval_seconds": 0.1,
    }
    with pytest.raises(RuntimeError, match="denied"):
        await xai._poll_for_tokens(device, None)


# ---------------------------------------------------------------------------
# Radius loopback 服务器（真实 socket）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_radius_loopback_server_receives_code() -> None:
    import httpx

    waiter = asyncio.create_task(radius._wait_for_browser_code("radius-state", asyncio.Event()))
    await asyncio.sleep(0.3)  # 服务器就绪
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(
            f"http://127.0.0.1:{radius.CALLBACK_PORT}{radius.CALLBACK_PATH}"
            "?code=radius-code&state=radius-state"
        )
    assert response.status_code == 200
    assert b"Signed in to Radius" in response.content
    code = await asyncio.wait_for(waiter, timeout=5)
    assert code == "radius-code"


@pytest.mark.asyncio
async def test_radius_loopback_state_mismatch_rejected() -> None:
    import httpx

    waiter = asyncio.create_task(radius._wait_for_browser_code("radius-state", asyncio.Event()))
    await asyncio.sleep(0.3)
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(
            f"http://127.0.0.1:{radius.CALLBACK_PORT}{radius.CALLBACK_PATH}?code=x&state=WRONG"
        )
    assert response.status_code == 400
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
