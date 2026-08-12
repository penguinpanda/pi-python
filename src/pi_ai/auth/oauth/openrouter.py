"""OpenRouter OAuth PKCE 流程（对齐 TS auth/oauth/openrouter.ts）。

OpenRouter 授权码换取的是永久 API key（refresh 返回原凭证）。
浏览器回调由一次性 loopback server（ephemeral 端口）接收，
并与手动粘贴回退竞争（浏览器无法触达本机时）。
"""

import asyncio
import sys
import urllib.parse
import uuid
import webbrowser

from typing import Any, cast

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .pkce import generate_pkce

_AsyncClient = httpx.AsyncClient

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/oauth/callback"
# 手动粘贴占位地址（loopback server 不可用时）。
FALLBACK_CALLBACK_URL = f"http://{CALLBACK_HOST}:1457{CALLBACK_PATH}"


def _parse_authorization_input(value: str | None) -> str | None:
    """从用户输入提取授权码（URL / code= 串 / 原始码）。"""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(trimmed)
        code = urllib.parse.parse_qs(parsed.query).get("code")
        return code[0] if code else None
    if "code=" in trimmed:
        params = urllib.parse.parse_qs(trimmed)
        code = params.get("code")
        return code[0] if code else None
    return trimmed


async def exchange_authorization_code(code: str, verifier: str) -> OAuthCredential:
    """授权码 → OpenRouter 永久 API key。"""
    async with _AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={
                "code": code,
                "code_verifier": verifier,
                "code_challenge_method": "S256",
            },
        )
    if not response.is_success:
        raise RuntimeError(f"OpenRouter OAuth key exchange failed (HTTP {response.status_code})")
    body = response.json()
    key = body.get("key") if isinstance(body, dict) else None
    if not isinstance(key, str) or not key:
        raise RuntimeError('OpenRouter OAuth response carries no "key"')
    return {
        "type": "oauth",
        "access": key,
        "refresh": "",
        "expires": sys.maxsize,
    }


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    verifier, challenge = generate_pkce()
    state = uuid.uuid4().hex
    callback_url = await _start_loopback_callback(verifier, interaction.signal)
    if callback_url is None:
        callback_url = FALLBACK_CALLBACK_URL
    query = urllib.parse.urlencode(
        {
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    url = f"{AUTHORIZE_URL}?{query}"
    interaction.notify(
        {
            "type": "auth_url",
            "url": url,
            "instructions": (
                "Complete sign-in in your browser. If the browser is on another "
                "machine, paste the final redirect URL here."
            ),
        }
    )
    try:
        webbrowser.open(url)
    except Exception:
        pass

    manual_prompt: Any = {
        "type": "manual_code",
        "message": (
            "Complete sign-in in your browser, or paste the authorization code / redirect URL here:"
        ),
        "placeholder": callback_url,
    }
    if callback_url != FALLBACK_CALLBACK_URL:
        # 浏览器回调与手动粘贴竞争（对齐 TS：任一先完成即用）。
        manual_task = asyncio.create_task(interaction.prompt(manual_prompt))
        callback_task = asyncio.create_task(_wait_for_callback_credential())
        done, _pending = await asyncio.wait(
            {manual_task, callback_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if callback_task in done and callback_task.result() is not None:
            manual_task.cancel()
            return cast(OAuthCredential, callback_task.result())
        manual_task.cancel()
        if callback_task in done and callback_task.result() is None:
            # 回调服务器失败且 manual 也完成：走 manual 结果。
            pass
        elif callback_task not in done:
            callback_task.cancel()
        manual = manual_task.result() if not manual_task.cancelled() else None
        if manual is None and callback_task in done:
            manual = None
        if manual is None:
            raise RuntimeError("OpenRouter login cancelled")
    else:
        manual = await interaction.prompt(manual_prompt)

    code = _parse_authorization_input(manual)
    if not code:
        raise RuntimeError("Missing authorization code")
    interaction.notify({"type": "progress", "message": "Exchanging authorization code..."})
    return await exchange_authorization_code(code, verifier)


_callback_credential: OAuthCredential | None = None
_callback_done: asyncio.Event = asyncio.Event()
_callback_server: asyncio.AbstractServer | None = None


async def _start_loopback_callback(verifier: str, signal: Any) -> str | None:
    """一次性 loopback 回调服务器（对齐 TS startCallbackServer）。"""
    global _callback_credential, _callback_done, _callback_server
    _callback_credential = None
    _callback_done = asyncio.Event()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        global _callback_credential
        request_line = await reader.readline()
        parts = request_line.decode("utf-8", "replace").split()
        path = parts[1] if len(parts) > 1 else "/"
        parsed = urllib.parse.urlparse(path)
        if parsed.path != CALLBACK_PATH:
            await _respond_openrouter(writer, 404, "OAuth callback route not found.")
            return
        query = urllib.parse.parse_qs(parsed.query)
        error = query.get("error", [None])[0]
        if error:
            description = query.get("error_description", [error])[0]
            await _respond_openrouter(
                writer, 400, f"OpenRouter authorization was denied: {description}"
            )
            _callback_done.set()
            return
        code = query.get("code", [None])[0]
        if not code:
            await _respond_openrouter(writer, 400, "OpenRouter returned no authorization code.")
            return
        try:
            credential = await exchange_authorization_code(code, verifier)
        except Exception as exc:
            await _respond_openrouter(writer, 500, f"Token exchange failed: {exc}")
            _callback_done.set()
            return
        await _respond_openrouter(
            writer, 200, "Signed in to OpenRouter. You may now close this page."
        )
        _callback_credential = credential
        _callback_done.set()

    try:
        server = await asyncio.start_server(_handle, CALLBACK_HOST, 0)
    except OSError:
        return None
    _callback_server = server

    async def _close_when_done() -> None:
        waiter = asyncio.create_task(_callback_done.wait())
        abort = asyncio.create_task(signal.wait()) if signal is not None else None
        tasks = {waiter} | ({abort} if abort is not None else set())
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            if not task.done():
                task.cancel()
        server.close()
        await server.wait_closed()

    asyncio.get_running_loop().create_task(_close_when_done())
    port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"


async def _wait_for_callback_credential() -> OAuthCredential | None:
    """等待浏览器回调完成交换（无响应时返回 None，走手动粘贴）。"""
    global _callback_credential, _callback_done
    try:
        await asyncio.wait_for(_callback_done.wait(), timeout=120)
    except asyncio.TimeoutError:
        return None
    return _callback_credential


async def _respond_openrouter(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    body = f"<html><body><p>{message}</p></body></html>".encode("utf-8")
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(
        status, "OK"
    )
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )
    await writer.drain()
    writer.close()


async def _refresh(credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
    # OpenRouter 的 key 是永久的，无 refresh 概念。
    return credential


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return {"api_key": credential["access"]}


class _OpenRouterOAuth:
    name = "OpenRouter OAuth"
    loginLabel = "Sign in with OpenRouter"
    login = staticmethod(_login)
    refresh = staticmethod(_refresh)
    to_auth = staticmethod(_to_auth)


open_router_oauth: OAuthAuth = _OpenRouterOAuth()  # type: ignore[assignment]


__all__ = ["open_router_oauth", "exchange_authorization_code", "_parse_authorization_input"]
