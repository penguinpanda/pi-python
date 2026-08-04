"""GitHub Copilot OAuth 设备代码流程（对齐 TS auth/oauth/github-copilot.ts）。

纯设备代码流程（无本地回调服务器）：GitHub OAuth → Copilot token，
从 token 的 proxy-ep 字段推导 API base URL，登录后启用已知模型。
"""

import base64
import re

from typing import Any

import httpx

from ..types import AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from .device_code import poll_oauth_device_code_flow

_AsyncClient = httpx.AsyncClient

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode("ascii")
COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"


def _normalize_domain(value: str) -> str | None:
    """规范化 GitHub Enterprise 域名输入；非法输入返回 None。"""
    trimmed = value.strip()
    if not trimmed:
        return None
    if "://" in trimmed:
        host = trimmed.split("://", 1)[1].split("/", 1)[0]
    else:
        host = trimmed.split("/", 1)[0]
    return host if "." in host else None


def _get_urls(domain: str) -> dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def get_base_url_from_token(token: str) -> str | None:
    """从 Copilot token 的 proxy-ep 字段推导 API base URL。"""
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = proxy_host.replace("proxy.", "api.", 1)
    return f"https://{api_host}"


def get_github_copilot_base_url(
    token: str | None = None,
    enterprise_domain: str | None = None,
) -> str:
    if token:
        url = get_base_url_from_token(token)
        if url:
            return url
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


async def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    async with _AsyncClient(timeout=timeout) as client:
        response = await client.request(method, url, headers=headers, data=data)
    if not response.is_success:
        raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {response.text}")
    return response.json()


async def start_device_flow(domain: str) -> dict[str, Any]:
    """GitHub 设备授权（RFC 8628）。"""
    data = await _fetch_json(
        _get_urls(domain)["device_code_url"],
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        data={"client_id": CLIENT_ID, "scope": "read:user"},
    )
    for field in ("device_code", "user_code", "verification_uri", "expires_in"):
        if field not in data:
            raise RuntimeError(f"Invalid device code response: missing {field}")
    return data


async def _poll_for_github_access_token(
    domain: str,
    device: dict[str, Any],
    signal: Any,
) -> str:
    urls = _get_urls(domain)

    async def poll() -> dict[str, Any]:
        raw = await _fetch_json(
            urls["access_token_url"],
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": device["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return {"status": "complete", "value": raw["access_token"]}
        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            if error == "authorization_pending":
                return {"status": "pending"}
            if error == "slow_down":
                interval = raw.get("interval")
                return {
                    "status": "slow_down",
                    "interval_seconds": interval if isinstance(interval, (int, float)) else None,
                }
            description = raw.get("error_description")
            suffix = f": {description}" if description else ""
            return {"status": "failed", "message": f"Device flow failed: {error}{suffix}"}
        return {"status": "failed", "message": "Invalid device token response"}

    return await poll_oauth_device_code_flow(
        poll,
        interval_seconds=device.get("interval"),
        expires_in_seconds=device.get("expires_in"),
        wait_before_first_poll=True,
        signal=signal,
    )


async def refresh_copilot_access_token(
    refresh_token: str,
    enterprise_domain: str | None = None,
) -> OAuthCredential:
    """用 GitHub access token 换取 Copilot token。"""
    domain = enterprise_domain or "github.com"
    raw = await _fetch_json(
        _get_urls(domain)["copilot_token_url"],
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {refresh_token}",
            **COPILOT_HEADERS,
        },
    )
    token = raw.get("token") if isinstance(raw, dict) else None
    expires_at = raw.get("expires_at") if isinstance(raw, dict) else None
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise RuntimeError("Invalid Copilot token response fields")
    credential: OAuthCredential = {
        "type": "oauth",
        "refresh": refresh_token,
        "access": token,
        "expires": int(expires_at * 1000) - 5 * 60 * 1000,
        "enterprise_url": enterprise_domain,
    }
    return credential


async def _login(interaction: AuthInteraction) -> OAuthCredential:
    prompt_result = await interaction.prompt(
        {
            "type": "text",
            "message": "GitHub Enterprise URL/domain (blank for github.com)",
            "placeholder": "company.ghe.com",
        }
    )
    trimmed = prompt_result.strip()
    enterprise_domain = _normalize_domain(trimmed) if trimmed else None
    if trimmed and enterprise_domain is None:
        raise RuntimeError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await start_device_flow(domain)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri"],
            "interval_seconds": device.get("interval"),
            "expires_in_seconds": device.get("expires_in"),
        }
    )
    github_access_token = await _poll_for_github_access_token(domain, device, interaction.signal)
    credential = await refresh_copilot_access_token(github_access_token, enterprise_domain or None)
    interaction.notify({"type": "progress", "message": "Signed in to GitHub Copilot"})
    return credential


async def _refresh(credential: OAuthCredential, signal: Any = None) -> OAuthCredential:
    return await refresh_copilot_access_token(
        credential["refresh"],
        credential.get("enterprise_url"),
    )


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return {
        "api_key": credential["access"],
        "base_url": get_github_copilot_base_url(
            credential.get("access"),
            credential.get("enterprise_url"),
        ),
    }


class _GitHubCopilotOAuth:
    name = "GitHub Copilot"
    login = staticmethod(_login)
    refresh = staticmethod(_refresh)
    to_auth = staticmethod(_to_auth)


github_copilot_oauth: OAuthAuth = _GitHubCopilotOAuth()  # type: ignore[assignment]


__all__ = [
    "github_copilot_oauth",
    "get_base_url_from_token",
    "get_github_copilot_base_url",
    "refresh_copilot_access_token",
]
