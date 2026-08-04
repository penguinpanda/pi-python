"""OpenRouter OAuth PKCE 流程（对齐 TS auth/oauth/openrouter.ts）。

OpenRouter 授权码换取的是永久 API key（refresh 返回原凭证）。
首批实现 manual paste（无本地回调服务器）：
用户打开授权 URL 授权后，粘贴最终重定向 URL / 授权码。
"""

import sys
import urllib.parse

from typing import Any

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .pkce import generate_pkce

_AsyncClient = httpx.AsyncClient

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
# 占位回调地址：仅用于承载 code 查询参数（浏览器访问会失败，但 URL 可复制）。
CALLBACK_URL = "http://127.0.0.1:1457/oauth/callback"


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
    query = urllib.parse.urlencode(
        {
            "callback_url": CALLBACK_URL,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
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
    manual = await interaction.prompt(
        {
            "type": "manual_code",
            "message": (
                "Complete sign-in in your browser, or paste the authorization "
                "code / redirect URL here:"
            ),
            "placeholder": CALLBACK_URL,
        }
    )
    code = _parse_authorization_input(manual)
    if not code:
        raise RuntimeError("Missing authorization code")
    interaction.notify({"type": "progress", "message": "Exchanging authorization code..."})
    return await exchange_authorization_code(code, verifier)


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
