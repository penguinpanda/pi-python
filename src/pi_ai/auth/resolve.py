"""认证解析（对齐 TS auth/resolve.ts）。

- resolve_provider_auth：API Key / OAuth 统一解析；
- resolve_stored_oauth：双重检查锁定 + minOAuthValidityMs，
  并发请求不会双刷新已轮换的 token。
"""

from typing import Any

from ..types.common import ProviderEnv, now_ms
from .context import AuthContext
from .types import (
    AuthResult,
    Credential,
    CredentialStore,
    ModelAuth,
    OAuthAuth,
    credential_type,
)

DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000


class ModelsError(Exception):
    """统一错误（code: model_source/model_validation/provider/stream/auth/oauth）。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if cause is not None:
            detail = str(cause).strip()
            if detail and detail not in message:
                message = f"{message}: {detail}"
        super().__init__(message)
        self.code = code
        self.cause = cause


def _credential_key(credential: Any) -> str | None:
    if isinstance(credential, dict):
        return credential.get("key")
    return getattr(credential, "key", None)


async def read_credential(
    credentials: CredentialStore,
    provider_id: str,
) -> Credential | None:
    try:
        return await credentials.read(provider_id)
    except Exception as exc:
        raise ModelsError(
            "auth", f"Credential store read failed for {provider_id}", cause=exc
        ) from exc


async def _resolve_api_key_legacy(
    ctx: AuthContext,
    env_auth: Any,
    provider_id: str,
    credential: Credential | None,
) -> AuthResult | None:
    """旧式 EnvApiKeyAuth 解析（stored key → env vars，经 AuthContext）。"""
    try:
        key = _credential_key(credential)
        if key:
            return AuthResult(auth={"api_key": key}, source="stored credential")
        for var in getattr(env_auth, "env_vars", []):
            value = await ctx.env(var)
            if value:
                return AuthResult(auth={"api_key": value}, source=var)
        return None
    except Exception as exc:
        raise ModelsError(
            "auth", f"API key auth failed for provider {provider_id}", cause=exc
        ) from exc


def _expires_soon(
    credential: dict[str, Any],
    minimum_validity_ms: int,
) -> bool:
    return now_ms() + minimum_validity_ms >= credential.get("expires", 0)


async def resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: dict[str, Any],
    min_oauth_validity_ms: int | None = None,
) -> AuthResult | None:
    """OAuth 解析：双重检查锁定 + 刷新 + 持久化（对齐 TS resolveStoredOAuth）。"""
    minimum_validity_ms = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)
    credential = stored

    if _expires_soon(credential, minimum_validity_ms):
        async def _refresh(current: Credential | None) -> Credential | None:
            if current is None or credential_type(current) != "oauth":
                return None  # 期间被 logout
            if not _expires_soon(current, minimum_validity_ms):
                return None  # 已被其它请求/进程刷新
            try:
                return await oauth.refresh(current)
            except Exception as exc:
                raise ModelsError(
                    "oauth", f"OAuth refresh failed for {provider_id}", cause=exc
                ) from exc

        try:
            post = await credentials.modify(provider_id, _refresh)
        except ModelsError:
            raise
        except Exception as exc:
            raise ModelsError(
                "auth",
                f"Credential store modify failed for {provider_id}",
                cause=exc,
            ) from exc

        if post is None or credential_type(post) != "oauth":
            return None  # 期间被 logout
        credential = post
        # 显式调用方要求刷新后仍有足够有效期。
        if min_oauth_validity_ms is not None and _expires_soon(
            credential, min_oauth_validity_ms
        ):
            raise ModelsError(
                "oauth",
                f"OAuth refresh returned a token that expires too soon for {provider_id}",
            )

    try:
        return AuthResult(auth=await oauth.to_auth(credential), source="OAuth")
    except Exception as exc:
        raise ModelsError(
            "oauth", f"OAuth auth derivation failed for {provider_id}", cause=exc
        ) from exc


async def resolve_provider_auth(
    provider: Any,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: dict[str, Any] | None = None,
) -> AuthResult | None:
    """解析 provider 认证：显式 api_key → 存储凭证 → 环境变量。

    provider 需有 .id 与 .auth（EnvApiKeyAuth 或带 oauth 的对象）。
    """
    provider_id = provider.id
    auth = getattr(provider, "auth", None)
    overrides = overrides or {}

    explicit_key = overrides.get("api_key")
    if explicit_key:
        if auth is not None and hasattr(auth, "resolve"):
            return await _resolve_api_key_legacy(
                auth_context,
                auth,
                provider_id,
                {"type": "api_key", "key": explicit_key},
            )
        return None

    stored = await read_credential(credentials, provider_id)
    if stored is not None:
        ctype = credential_type(stored)
        if ctype == "oauth":
            oauth = getattr(auth, "oauth", None)
            if oauth is None:
                return None
            return await resolve_stored_oauth(
                credentials,
                provider_id,
                oauth,
                stored,
                overrides.get("min_oauth_validity_ms"),
            )
        if ctype == "api_key" and auth is not None and hasattr(auth, "resolve"):
            return await _resolve_api_key_legacy(auth_context, auth, provider_id, stored)
        return None

    # 环境变量（ambient）。
    if auth is not None and hasattr(auth, "resolve"):
        return await _resolve_api_key_legacy(auth_context, auth, provider_id, None)
    return None


__all__ = [
    "ModelsError",
    "resolve_provider_auth",
    "resolve_stored_oauth",
    "read_credential",
    "DEFAULT_OAUTH_MINIMUM_VALIDITY_MS",
]
