"""Radius 网关 OAuth 浏览器流程（对齐 TS auth/oauth/radius.ts）。

网关自发现 authorization endpoint；PKCE + 固定端口 loopback 回调；
token 经网关 /v1/oauth/token 交换。
"""

from __future__ import annotations

import asyncio
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
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_SECONDS = 60
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
DEFAULT_AUTHORIZATION_ENDPOINT = "https://radius.pi.dev/oauth/authorize"


def _normalize_gateway(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value.rstrip("/")


async def _discover_authorization_endpoint(gateway: str) -> str:
    """发现授权端点（对齐 TS discoverAuthorizationEndpoint 语义）。"""
    try:
        async with _AsyncClient(timeout=15) as client:
            response = await client.get(f"{gateway}/.well-known/oauth-authorization-server")
            if response.is_success:
                data = response.json()
                if isinstance(data, dict) and isinstance(data.get("authorization_endpoint"), str):
                    return data["authorization_endpoint"]
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
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("access_token"), str)
        or not isinstance(data.get("refresh_token"), str)
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


async def _wait_for_browser_code(state: str, signal: Any) -> str | None:
    """固定端口 loopback 回调服务器（对齐 TS startOAuthCallbackServer）。"""
    result: dict = {"code": None}
    completed = asyncio.Event()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = await reader.readline()
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
    body = f"<html><body><p>{message}</p></body></html>".encode("utf-8")
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )
    await writer.drain()
    writer.close()


async def _login(interaction: AuthInteraction, gateway: str = "") -> OAuthCredential:
    gateway = _normalize_gateway(gateway or getattr(interaction, "gateway", "") or "")
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
        pasted = manual_task.result() if not manual_task.cancelled() else None
        if pasted is None:
            raise RuntimeError("Radius OAuth login was cancelled")
        parsed = urllib.parse.urlparse(pasted.strip())
        if parsed.scheme:
            code = urllib.parse.parse_qs(parsed.query).get("code", [None])[0]
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
        _normalize_gateway(str(gateway) if gateway else ""),
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
            return await _login(interaction, gateway)

        async def refresh(self, credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
            return await _request_oauth_token(
                _normalize_gateway(gateway),
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
