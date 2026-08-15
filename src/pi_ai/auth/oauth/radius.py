"""Radius 网关 OAuth 浏览器流程（对齐 TS auth/oauth/radius.ts）。

网关自发现 authorization endpoint；PKCE + 固定端口 loopback 回调；
token 经网关 /v1/oauth/token 交换。
"""

from __future__ import annotations

import asyncio
import html
import time
import urllib.parse
import uuid
import webbrowser

from typing import Any, cast

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .pkce import generate_pkce

_AsyncClient = httpx.AsyncClient

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1456
CALLBACK_PATH = "/oauth/callback"
# 回调请求读取超时：防止慢连接/不发数据的连接挂起 handler 任务。
CALLBACK_READ_TIMEOUT = 10.0
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_SECONDS = 60
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device-code"
DEFAULT_AUTHORIZATION_ENDPOINT = "https://radius.pi.dev/oauth/authorize"


def normalize_gateway_url(value: str) -> str:
    """补 https:// 并去尾斜杠；非回环地址强制 https。

    显式 http:// 的网关会把 refresh/access token 明文暴露给局域网嗅探，
    仅允许 127.0.0.1 / localhost / ::1 使用明文（本地开发）。
    """
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    normalized = value.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    host = parsed.hostname or ""
    if parsed.scheme != "https" and host not in ("127.0.0.1", "localhost", "::1"):
        normalized = "https://" + normalized[len("http://") :]
    return normalized


async def _discover_authorization_endpoint(gateway: str) -> str:
    """发现授权端点（对齐 TS：GET {gateway}/v1/oauth）。

    兼容两种发现响应键（authorizationEndpoint / authorization_endpoint），
    并校验 scheme/host 与网关一致，防止恶意网关把授权端点指向任意 URL。
    """
    try:
        async with _AsyncClient(timeout=15) as client:
            response = await client.get(f"{gateway}/v1/oauth")
            if response.is_success:
                data = response.json()
                if isinstance(data, dict):
                    endpoint = data.get("authorizationEndpoint") or data.get(
                        "authorization_endpoint"
                    )
                    if isinstance(endpoint, str):
                        parsed_endpoint = urllib.parse.urlparse(endpoint)
                        gateway_host = urllib.parse.urlparse(gateway).hostname
                        is_loopback = parsed_endpoint.hostname in ("127.0.0.1", "localhost", "::1")
                        if (
                            parsed_endpoint.scheme in ("https", "http")
                            and parsed_endpoint.hostname == gateway_host
                            and (parsed_endpoint.scheme == "https" or is_loopback)
                        ):
                            return endpoint
    except (httpx.HTTPError, ValueError):
        pass
    return f"{gateway}/oauth/authorize"


async def _request_oauth_token(
    gateway: str, fields: dict[str, str], signal: Any
) -> OAuthCredential:
    try:
        async with _AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{gateway}/v1/oauth/token",
                headers={
                    "accept": "application/json",
                    "content-type": "application/x-www-form-urlencoded",
                },
                content=urllib.parse.urlencode(fields),
            )
    except httpx.HTTPError as exc:
        if signal is not None and signal.is_set():
            raise RuntimeError("Login cancelled") from exc
        raise
    if not response.is_success:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("error_description") or body.get("error") or "")
        except ValueError:
            detail = response.text[:200]
        raise RuntimeError(
            f"Radius OAuth token request failed (HTTP {response.status_code})"
            f"{': ' + detail if detail else ''}"
        )
    data = response.json()
    return (
        _parse_oauth_token_response(data)
        if isinstance(data, dict)
        else _parse_oauth_token_response({})
    )


async def _wait_for_browser_code(state: str, signal: Any) -> str | None:
    """固定端口 loopback 回调服务器（对齐 TS startOAuthCallbackServer）。"""
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
        parsed = urllib.parse.urlparse(path)
        if parsed.path != CALLBACK_PATH:
            await _respond(writer, 404, "Callback route not found.")
            return
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("state", [""])[0] != state:
            await _respond(writer, 400, "OAuth state mismatch.")
            return
        error = query.get("error", [None])[0]
        if error:
            description = query.get("error_description", [error])[0]
            await _respond(writer, 400, description)
            completed.set()
            return
        code = query.get("code", [None])[0]
        if not code:
            await _respond(writer, 400, "Missing authorization code.")
            return
        await _respond(writer, 200, "Signed in to Radius. You may now close this page.")
        result["code"] = code
        completed.set()

    try:
        server = await asyncio.start_server(_handle, CALLBACK_HOST, CALLBACK_PORT)
    except OSError:
        return None
    try:
        waiter = asyncio.create_task(completed.wait())
        abort = asyncio.create_task(signal.wait()) if signal is not None else None
        tasks = {waiter} | ({abort} if abort is not None else set())
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        server.close()
        await server.wait_closed()
    return result["code"]


async def _respond(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    # error_description 等来自回调 query 的攻击者可控文本,必须转义。
    body = f"<html><body><p>{html.escape(message)}</p></body></html>".encode("utf-8")
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )
    await writer.drain()
    writer.close()


async def _request_device_authorization(gateway: str, signal: Any) -> dict[str, Any]:
    """Radius /v1/oauth/device 设备授权（对齐 TS requestDeviceAuthorization）。"""
    try:
        async with _AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{gateway}/v1/oauth/device",
                headers={
                    "accept": "application/json",
                    "content-type": "application/x-www-form-urlencoded",
                },
                content=urllib.parse.urlencode(
                    {"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE}
                ),
            )
    except httpx.HTTPError as exc:
        if signal is not None and signal.is_set():
            raise RuntimeError("Login cancelled") from exc
        raise
    if not response.is_success:
        raise RuntimeError(
            f"Radius OAuth device authorization failed (HTTP {response.status_code})"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Radius OAuth device authorization response is invalid")
    for field in ("device_code", "user_code", "verification_uri", "expires_in"):
        if not data.get(field):
            raise RuntimeError(
                "Radius OAuth device authorization response is missing required fields"
            )
    return data


async def _poll_device_token(gateway: str, device: dict[str, Any], signal: Any) -> OAuthCredential:
    from .device_code import poll_oauth_device_code_flow

    interval = device.get("interval")
    interval_seconds = (
        float(interval)
        if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0
        else None
    )

    async def poll() -> dict[str, Any]:
        try:
            async with _AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{gateway}/v1/oauth/token",
                    headers={
                        "accept": "application/json",
                        "content-type": "application/x-www-form-urlencoded",
                    },
                    content=urllib.parse.urlencode(
                        {
                            "grant_type": OAUTH_DEVICE_CODE_GRANT_TYPE,
                            "client_id": OAUTH_CLIENT_ID,
                            "device_code": device["device_code"],
                        }
                    ),
                )
        except httpx.HTTPError as exc:
            if signal is not None and signal.is_set():
                raise RuntimeError("Login cancelled") from exc
            raise
        if response.is_success:
            data = response.json()
            if not isinstance(data, dict):
                return {"status": "failed", "message": "Invalid Radius token response"}
            try:
                value = _parse_oauth_token_response(data)
            except RuntimeError as exc:
                return {"status": "failed", "message": str(exc)}
            return {"status": "complete", "value": value}
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if error == "authorization_pending":
            return {"status": "pending"}
        if error == "slow_down":
            raw_interval = body.get("interval")
            return {
                "status": "slow_down",
                "interval_seconds": raw_interval
                if isinstance(raw_interval, (int, float)) and not isinstance(raw_interval, bool)
                else None,
            }
        if error == "expired_token":
            return {"status": "failed", "message": "Radius device code expired"}
        if error in ("access_denied", "authorization_denied"):
            return {"status": "failed", "message": "Radius device authorization was denied"}
        return {
            "status": "failed",
            "message": f"Radius OAuth device token polling failed (HTTP {response.status_code})",
        }

    return await poll_oauth_device_code_flow(
        poll,
        interval_seconds=interval_seconds,
        expires_in_seconds=int(device.get("expires_in") or 0),
        wait_before_first_poll=True,
        signal=signal,
    )


def _parse_oauth_token_response(data: dict[str, Any]) -> OAuthCredential:
    if not isinstance(data.get("access_token"), str) or not isinstance(
        data.get("refresh_token"), str
    ):
        raise RuntimeError("Radius OAuth token response missing fields")
    credential: OAuthCredential = {
        "type": "oauth",
        "access": data["access_token"],
        "refresh": data["refresh_token"],
        "expires": int(time.time() * 1000)
        + int(data.get("expires_in", 0) or 0) * 1000
        - TOKEN_EXPIRY_SKEW_SECONDS * 1000,
    }
    if isinstance(data.get("scope"), str):
        credential["scope"] = data["scope"]  # type: ignore[typeddict-unknown-key]
    return credential


async def _login_device_code(interaction: AuthInteraction, gateway: str) -> OAuthCredential:
    device = await _request_device_authorization(gateway, interaction.signal)
    interval = device.get("interval")
    interaction.notify(
        cast(
            Any,
            {
                "type": "device_code",
                "user_code": device["user_code"],
                "verification_uri": device["verification_uri"],
                "interval_seconds": int(interval) if isinstance(interval, (int, float)) else None,
                "expires_in_seconds": int(device.get("expires_in") or 0),
            },
        )
    )
    return await _poll_device_token(gateway, device, interaction.signal)


async def _login(interaction: AuthInteraction, gateway: str = "") -> OAuthCredential:
    gateway = normalize_gateway_url(gateway or getattr(interaction, "gateway", "") or "")
    verifier, challenge = generate_pkce()
    state = uuid.uuid4().hex
    authorize_endpoint = await _discover_authorization_endpoint(gateway)
    url = f"{authorize_endpoint}?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": OAUTH_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    interaction.notify(
        {
            "type": "auth_url",
            "url": url,
            "instructions": "A browser window should open. Complete login to finish.",
        }
    )
    try:
        webbrowser.open(url)
    except Exception:
        pass

    manual_prompt: Any = {
        "type": "manual_code",
        "message": "Complete login in your browser, or paste the authorization "
        "code / redirect URL here:",
        "placeholder": REDIRECT_URI,
    }
    # 浏览器回调与手动粘贴竞争（对齐 TS：任一先完成即用，避免挂起）。
    manual_task = asyncio.create_task(interaction.prompt(manual_prompt))
    callback_task = asyncio.create_task(_wait_for_browser_code(state, interaction.signal))
    done, _pending = await asyncio.wait(
        {manual_task, callback_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if callback_task in done and callback_task.result():
        manual_task.cancel()
        await asyncio.gather(manual_task, return_exceptions=True)
        code = callback_task.result()
    else:
        if callback_task in done:
            await asyncio.gather(callback_task, return_exceptions=True)
        else:
            callback_task.cancel()
            await asyncio.gather(callback_task, return_exceptions=True)
        # callback 未提供 code：等待手动粘贴（避免对 pending 任务调
        # result() 抛 InvalidStateError）。
        pasted = await manual_task
        if pasted is None:
            raise RuntimeError("Radius OAuth login was cancelled")
        parsed = urllib.parse.urlparse(pasted.strip())
        if parsed.scheme:
            query = urllib.parse.parse_qs(parsed.query)
            pasted_state = query.get("state", [None])[0]
            if pasted_state is not None and pasted_state != state:
                # 粘贴 URL 的 state 与当前 flow 不一致：拒绝旧登录 code。
                raise RuntimeError("Radius OAuth login: pasted URL state mismatch")
            code = query.get("code", [None])[0]
        else:
            code = pasted.strip() or None
    if not code:
        raise RuntimeError("Radius OAuth login was cancelled")
    interaction.notify({"type": "progress", "message": "Exchanging authorization code..."})
    return await _request_oauth_token(
        gateway,
        {
            "grant_type": "authorization_code",
            "client_id": OAUTH_CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        },
        interaction.signal,
    )


async def _refresh(credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
    gateway = credential.get("gateway") or ""
    return await _request_oauth_token(
        normalize_gateway_url(str(gateway) if gateway else ""),
        {
            "grant_type": "refresh_token",
            "client_id": OAUTH_CLIENT_ID,
            "refresh_token": credential["refresh"],
        },
        signal,
    )


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return {"api_key": credential["access"]}


def create_radius_oauth(gateway: str) -> OAuthAuth:
    """带网关参数的 Radius OAuth（对齐 TS lazyOAuth({name, gateway})）。"""

    class _BoundRadiusOAuth:
        name = "Radius (subscription)"
        is_subscription = True
        loginLabel = "Sign in with Radius"

        async def login(self, interaction: AuthInteraction) -> OAuthCredential:
            login_method = await interaction.prompt(
                cast(
                    Any,
                    {
                        "type": "select",
                        "message": "Sign in to Radius:",
                        "options": [
                            {
                                "id": LOGIN_METHOD_BROWSER,
                                "label": "Sign in with browser (recommended)",
                            },
                            {
                                "id": LOGIN_METHOD_DEVICE_CODE,
                                "label": "Sign in with device code (when signing in from another device)",
                            },
                        ],
                    },
                )
            )
            if login_method == LOGIN_METHOD_DEVICE_CODE:
                return await _login_device_code(interaction, gateway)
            if login_method == LOGIN_METHOD_BROWSER:
                return await _login(interaction, gateway)
            raise RuntimeError(f"Unknown Radius sign-in method: {login_method}")

        async def refresh(self, credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
            return await _request_oauth_token(
                normalize_gateway_url(gateway),
                {
                    "grant_type": "refresh_token",
                    "client_id": OAUTH_CLIENT_ID,
                    "refresh_token": credential["refresh"],
                },
                signal,
            )

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return await _to_auth(credential)

    return cast(OAuthAuth, _BoundRadiusOAuth())


__all__ = ["create_radius_oauth"]
