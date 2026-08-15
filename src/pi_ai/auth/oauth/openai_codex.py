"""OpenAI Codex（ChatGPT OAuth）登录（对齐 TS openai-codex.ts）。

优先浏览器流程（PKCE + 本地回调服务器 + 手动粘贴回退）；
设备代码流程作为兼容路径。
"""

import asyncio
import base64
import json
import time
import uuid
import webbrowser

from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .device_code import poll_oauth_device_code_flow
from .pkce import generate_pkce

_AsyncClient = httpx.AsyncClient

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
BROWSER_REDIRECT_URI = "http://localhost:1455/auth/callback"
BROWSER_SCOPE = "openid profile email offline_access"
BROWSER_PORT = 1455
# 回调请求读取超时：防止慢连接/不发数据的连接挂起 handler 任务。
CALLBACK_READ_TIMEOUT = 10.0
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
    try:
        interval_seconds = (
            float(str(interval).strip()) if isinstance(interval, (int, float, str)) else None
        )
    except ValueError as exc:
        raise RuntimeError(f"Invalid OpenAI Codex device code response: {data}") from exc
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
            f"OpenAI Codex token exchange failed ({response.status_code}): {response.text[:500]}"
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
            f"OpenAI Codex token refresh failed ({response.status_code}): {response.text[:500]}"
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


def create_authorization_flow(originator: str = "pi") -> dict[str, str]:
    """PKCE 授权 URL（对齐 TS createAuthorizationFlow）。"""
    verifier, challenge = generate_pkce()
    state = uuid.uuid4().hex
    url = f"{AUTHORIZE_URL}?" + urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": BROWSER_REDIRECT_URI,
            "scope": BROWSER_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    return {"verifier": verifier, "state": state, "url": url}


async def _wait_for_browser_code(state: str, signal: Any) -> dict | None:
    """本地回调服务器（对齐 TS startLocalOAuthServer；固定端口 1455）。"""
    result: dict = {"code": None}
    completed = asyncio.Event()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=CALLBACK_READ_TIMEOUT)
        except asyncio.TimeoutError:
            writer.close()
            return
        parts = request_line.decode("utf-8", "replace").split()
        path = parts[1] if len(parts) > 1 else "/"
        parsed = urlparse(path)
        if parsed.path != "/auth/callback":
            _respond_html(writer, 404, "Callback route not found.")
            return
        query = parse_qs(parsed.query)
        if query.get("state", [""])[0] != state:
            _respond_html(writer, 400, "State mismatch.")
            return
        code = query.get("code", [""])[0]
        if not code:
            _respond_html(writer, 400, "Missing authorization code.")
            return
        _respond_html(writer, 200, "OpenAI authentication completed. You can close this window.")
        result["code"] = code
        completed.set()

    try:
        server = await asyncio.start_server(_handle, "127.0.0.1", BROWSER_PORT)
    except OSError:
        return None
    try:
        waiter = asyncio.create_task(completed.wait())
        abort = asyncio.create_task(signal.wait()) if signal is not None else None
        try:
            if abort is not None:
                done, _pending = await asyncio.wait(
                    {waiter, abort}, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, _pending = await asyncio.wait({waiter})
            for task in _pending if abort is not None else ():
                task.cancel()
        finally:
            if not waiter.done():
                waiter.cancel()
            if abort is not None and not abort.done():
                abort.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            if abort is not None:
                await asyncio.gather(abort, return_exceptions=True)
    finally:
        server.close()
        await server.wait_closed()
    return result if result["code"] else None


def _respond_html(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    body = f"<html><body><p>{message}</p></body></html>".encode("utf-8")
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )

    async def _finish() -> None:
        try:
            await writer.drain()
        finally:
            writer.close()

    asyncio.get_running_loop().create_task(_finish())


def _extract_code_from_pasted(value: str) -> tuple[str | None, str | None]:
    """从手动粘贴的授权 URL 提取 (code, state)（对齐 TS 手动粘贴回退）。"""
    try:
        parsed = urlparse(value.strip())
        if parsed.scheme:
            query = parse_qs(parsed.query)
            return query.get("code", [None])[0], query.get("state", [None])[0]
    except Exception:
        pass
    return None, None


async def _browser_login(interaction: AuthInteraction) -> OAuthCredential:
    flow = create_authorization_flow()
    interaction.notify(
        {
            "type": "auth_url",
            "url": flow["url"],
            "instructions": "A browser window should open. Complete login to finish.",
        }
    )
    try:
        webbrowser.open(flow["url"])
    except Exception:
        pass

    manual_prompt: Any = {
        "type": "manual_code",
        "message": "Complete login in your browser, or paste the authorization "
        "code / redirect URL here:",
        "placeholder": BROWSER_REDIRECT_URI,
    }
    # 浏览器回调与手动粘贴竞争（对齐 TS：任一先完成即用，避免挂起）。
    manual_task = asyncio.create_task(interaction.prompt(manual_prompt))
    callback_task = asyncio.create_task(_wait_for_browser_code(flow["state"], interaction.signal))
    done, _pending = await asyncio.wait(
        {manual_task, callback_task}, return_when=asyncio.FIRST_COMPLETED
    )
    server_result = callback_task.result() if callback_task in done else None
    if server_result is not None and server_result.get("code"):
        manual_task.cancel()
        await asyncio.gather(manual_task, return_exceptions=True)
        code = server_result["code"]
    else:
        if callback_task in done:
            await asyncio.gather(callback_task, return_exceptions=True)
        else:
            callback_task.cancel()
            await asyncio.gather(callback_task, return_exceptions=True)
        pasted = manual_task.result() if not manual_task.cancelled() else None
        if pasted is None:
            raise RuntimeError("OpenAI Codex browser login was cancelled")
        pasted_code, pasted_state = _extract_code_from_pasted(pasted)
        if pasted_state is not None and pasted_state != flow["state"]:
            raise RuntimeError("OpenAI Codex login failed: state mismatch in pasted URL")
        code = pasted_code or pasted.strip()
    if not code:
        raise RuntimeError("OpenAI Codex browser login was cancelled")
    token = await exchange_authorization_code(code, flow["verifier"], BROWSER_REDIRECT_URI)
    return _credentials_from_token(token)


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    try:
        device = await start_device_auth(interaction.signal)
    except RuntimeError as exc:
        if "browser login" in str(exc):
            return await _browser_login(interaction)
        raise
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
