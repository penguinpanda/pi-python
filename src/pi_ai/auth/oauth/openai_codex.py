"""OpenAI Codex（ChatGPT OAuth）设备代码流程（对齐 TS openai-codex.ts）。

首批只实现设备代码流程（无本地回调服务器）；浏览器流程后续按需补充。
"""

import base64
import json
import time

from typing import Any

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .device_code import poll_oauth_device_code_flow

_AsyncClient = httpx.AsyncClient

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


def get_account_id(access_token: str) -> str | None:
    payload = _decode_jwt_payload(access_token)
    if not payload:
        return None
    auth = payload.get(JWT_CLAIM_PATH)
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return account_id if isinstance(account_id, str) and account_id else None


async def start_device_auth(signal: Any = None) -> dict[str, Any]:
    async with _AsyncClient(timeout=30) as client:
        response = await client.post(
            DEVICE_USER_CODE_URL,
            headers={"Content-Type": "application/json"},
            json={"client_id": CLIENT_ID},
        )
    if response.status_code == 404:
        raise RuntimeError(
            "OpenAI Codex device code login is not enabled for this server. "
            "Use browser login or verify the server URL."
        )
    if not response.is_success:
        raise RuntimeError(
            f"OpenAI Codex device code request failed with status {response.status_code}"
        )
    data = response.json()
    interval = data.get("interval")
    interval_seconds = (
        float(str(interval).strip()) if isinstance(interval, (int, float, str)) else None
    )
    if (
        not isinstance(data.get("device_auth_id"), str)
        or not isinstance(data.get("user_code"), str)
        or not isinstance(interval_seconds, (int, float))
        or interval_seconds < 0
    ):
        raise RuntimeError(f"Invalid OpenAI Codex device code response: {data}")
    return {
        "device_auth_id": data["device_auth_id"],
        "user_code": data["user_code"],
        "interval_seconds": interval_seconds,
    }


async def _poll_device_auth(
    device: dict[str, Any],
    signal: Any,
) -> dict[str, str]:
    async def poll() -> dict[str, Any]:
        async with _AsyncClient(timeout=30) as client:
            response = await client.post(
                DEVICE_TOKEN_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "device_auth_id": device["device_auth_id"],
                    "user_code": device["user_code"],
                },
            )
        if response.is_success:
            data = response.json()
            if isinstance(data.get("authorization_code"), str) and isinstance(
                data.get("code_verifier"), str
            ):
                return {
                    "status": "complete",
                    "value": {
                        "authorization_code": data["authorization_code"],
                        "code_verifier": data["code_verifier"],
                    },
                }
            return {"status": "failed", "message": "Invalid device auth token response"}
        if response.status_code in (403, 404):
            return {"status": "pending"}
        try:
            error_code = (response.json() or {}).get("error")
        except Exception:
            error_code = None
        if isinstance(error_code, dict):
            error_code = error_code.get("code")
        if error_code == "deviceauth_authorization_pending":
            return {"status": "pending"}
        if error_code == "slow_down":
            return {"status": "slow_down"}
        return {
            "status": "failed",
            "message": f"OpenAI Codex device auth failed with status {response.status_code}",
        }

    return await poll_oauth_device_code_flow(
        poll,
        interval_seconds=device.get("interval_seconds"),
        expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        signal=signal,
    )


async def exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str = DEVICE_REDIRECT_URI,
) -> dict[str, Any]:
    """授权码 → access/refresh token。"""
    async with _AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
        )
    if not response.is_success:
        raise RuntimeError(
            f"OpenAI Codex token exchange failed ({response.status_code}): {response.text}"
        )
    data = response.json()
    if not isinstance(data.get("access_token"), str) or not isinstance(
        data.get("refresh_token"), str
    ):
        raise RuntimeError("OpenAI Codex token exchange response missing fields")
    return data


async def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    async with _AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
        )
    if not response.is_success:
        raise RuntimeError(
            f"OpenAI Codex token refresh failed ({response.status_code}): {response.text}"
        )
    data = response.json()
    if not isinstance(data.get("access_token"), str) or not isinstance(
        data.get("refresh_token"), str
    ):
        raise RuntimeError("OpenAI Codex token refresh response missing fields")
    return data


def _credentials_from_token(token: dict[str, Any]) -> OAuthCredential:
    access = token["access_token"]
    account_id = get_account_id(access)
    credential: OAuthCredential = {
        "type": "oauth",
        "access": access,
        "refresh": token["refresh_token"],
        "expires": int(time.time() * 1000) + int(token.get("expires_in", 0)) * 1000,
    }
    if account_id:
        credential["account_id"] = account_id
    return credential


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    device = await start_device_auth(interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": f"{AUTH_BASE_URL}/codex/device",
            "interval_seconds": int(device.get("interval_seconds", 5)),
            "expires_in_seconds": DEVICE_CODE_TIMEOUT_SECONDS,
        }
    )
    result = await _poll_device_auth(device, interaction.signal)
    token = await exchange_authorization_code(
        result["authorization_code"],
        result["code_verifier"],
        DEVICE_REDIRECT_URI,
    )
    return _credentials_from_token(token)


async def _refresh(credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
    token = await refresh_access_token(credential["refresh"])
    return _credentials_from_token(token)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return {"api_key": credential["access"]}


class _OpenAICodexOAuth:
    name = "OpenAI (ChatGPT Plus/Pro)"
    login = staticmethod(_login)
    refresh = staticmethod(_refresh)
    to_auth = staticmethod(_to_auth)


openai_codex_oauth: OAuthAuth = _OpenAICodexOAuth()  # type: ignore[assignment]


__all__ = [
    "openai_codex_oauth",
    "get_account_id",
    "exchange_authorization_code",
    "refresh_access_token",
]
