"""xAI OAuth device-code 流程（对齐 TS auth/oauth/xai.ts）。"""

from __future__ import annotations

import time
import urllib.parse

from typing import Any, cast

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .device_code import poll_oauth_device_code_flow

_AsyncClient = httpx.AsyncClient

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
# 到期前提前刷新，避免 token 在请求中途失效（对齐 TS REFRESH_SKEW_MS）。
REFRESH_SKEW_SECONDS = 5 * 60
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


def _required_string(body: dict, field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: dict, field: str) -> float:
    value = body.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return float(value)


def _validate_verification_uri(raw: str) -> str:
    """verification URI 强制 https（对齐 TS validateVerificationUri）。"""
    try:
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme != "https":
            raise ValueError("not https")
    except ValueError as exc:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response") from exc
    return raw


async def _post_form(url: str, fields: dict[str, str], signal: Any = None) -> httpx.Response:
    try:
        async with _AsyncClient(timeout=30) as client:
            return await client.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                content=urllib.parse.urlencode(fields),
            )
    except httpx.HTTPError as exc:
        if signal is not None and signal.is_set():
            raise RuntimeError("Login cancelled") from exc
        raise


def _credentials_from_token(
    body: dict, previous_refresh_token: str | None = None
) -> OAuthCredential:
    access = _required_string(body, "access_token")
    # 刷新时 xAI 可能不返回新 refresh_token（沿用旧值）。
    if body.get("refresh_token") is None and previous_refresh_token:
        refresh = previous_refresh_token
    else:
        refresh = _required_string(body, "refresh_token")
    if body.get("expires_in") is None:
        expires_in: float = DEFAULT_TOKEN_LIFETIME_SECONDS
    else:
        expires_in = _positive_number(body, "expires_in")
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000) + int(expires_in * 1000) - REFRESH_SKEW_SECONDS * 1000,
    }


async def _request_device_code(signal: Any = None) -> dict[str, Any]:
    response = await _post_form(
        XAI_DEVICE_CODE_URL,
        {
            "client_id": XAI_CLIENT_ID,
            "scope": XAI_SCOPE,
            "referrer": "pi",
        },
        signal,
    )
    if not response.is_success:
        raise RuntimeError(f"xAI OAuth device authorization failed (HTTP {response.status_code})")
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("xAI OAuth returned invalid JSON")
    interval = body.get("interval")
    interval_seconds = (
        float(interval)
        if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0
        else None
    )
    verification_uri_complete = body.get("verification_uri_complete")
    verification_uri = _validate_verification_uri(_required_string(body, "verification_uri"))
    return {
        "device_code": _required_string(body, "device_code"),
        "user_code": _required_string(body, "user_code"),
        "verification_uri": (
            _validate_verification_uri(verification_uri_complete)
            if isinstance(verification_uri_complete, str) and verification_uri_complete
            else verification_uri
        ),
        "interval_seconds": interval_seconds,
        "expires_in_seconds": _positive_number(body, "expires_in"),
    }


async def _poll_for_tokens(device: dict[str, Any], signal: Any) -> OAuthCredential:
    async def poll() -> dict[str, Any]:
        response = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": device["device_code"],
            },
            signal,
        )
        if response.is_success:
            body = response.json()
            if isinstance(body, dict):
                return {"status": "complete", "value": _credentials_from_token(body)}
            return {"status": "failed", "message": "Invalid xAI token response"}
        try:
            body = response.json() or {}
        except Exception:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if error == "authorization_pending":
            return {"status": "pending"}
        if error == "slow_down":
            interval = body.get("interval")
            return {
                "status": "slow_down",
                "interval_seconds": interval
                if isinstance(interval, (int, float)) and not isinstance(interval, bool)
                else None,
            }
        if error in ("access_denied", "authorization_denied"):
            return {"status": "failed", "message": "xAI device authorization was denied"}
        if error == "expired_token":
            return {"status": "failed", "message": "xAI device code expired"}
        return {
            "status": "failed",
            "message": f"xAI OAuth device token polling failed (HTTP {response.status_code})",
        }

    result = await poll_oauth_device_code_flow(
        poll,
        interval_seconds=device.get("interval_seconds"),
        expires_in_seconds=device["expires_in_seconds"],
        wait_before_first_poll=True,
        signal=signal,
    )
    return result


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    device = await _request_device_code(interaction.signal)
    interval_seconds = device.get("interval_seconds")
    interaction.notify(
        cast(
            Any,
            {
                "type": "device_code",
                "user_code": device["user_code"],
                "verification_uri": device["verification_uri"],
                "interval_seconds": int(interval_seconds) if interval_seconds is not None else None,
                "expires_in_seconds": device["expires_in_seconds"],
            },
        )
    )
    return await _poll_for_tokens(device, interaction.signal)


async def _refresh(credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
    response = await _post_form(
        XAI_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": XAI_CLIENT_ID,
            "refresh_token": credential["refresh"],
        },
        signal,
    )
    if not response.is_success:
        raise RuntimeError(f"xAI OAuth token refresh failed (HTTP {response.status_code})")
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("xAI OAuth returned invalid JSON")
    return _credentials_from_token(body, credential["refresh"])


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return {"api_key": credential["access"]}


class _XaiOAuth:
    name = "xAI (Grok/X subscription)"
    loginLabel = "Sign in with SuperGrok or X Premium"
    is_subscription = True
    login = staticmethod(_login)
    refresh = staticmethod(_refresh)
    to_auth = staticmethod(_to_auth)


xai_oauth: OAuthAuth = _XaiOAuth()  # type: ignore[assignment]

__all__ = ["xai_oauth"]
