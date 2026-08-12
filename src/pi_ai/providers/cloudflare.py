"""Cloudflare Workers AI / AI Gateway provider（自定义 resolve_auth）。"""

from __future__ import annotations

import os

from typing import Any, cast

from pi_ai.auth.types import AuthContext, AuthResult, CredentialStore
from pi_ai.provider import Provider, RefreshModelsContext, create_provider
from .openai_completions_providers import _fetch_openai_models


class _CloudflareAuth:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    async def resolve_auth(
        self,
        store: CredentialStore,
        ctx: AuthContext,
        options: dict[str, Any],
    ) -> AuthResult | None:
        credential: Any = None
        if store is not None:
            credential = await store.read("cloudflare-workers-ai")
            if credential is None:
                credential = await store.read("cloudflare-ai-gateway")
        credential_env = credential.get("env", {}) if isinstance(credential, dict) else {}
        credential_key = (
            credential.get("key")
            if isinstance(credential, dict)
            else getattr(credential, "key", None)
        )
        token = options.get("api_key") or credential_key or await ctx.env("CLOUDFLARE_API_KEY")
        account = (
            options.get("cloudflare_account_id")
            or credential_env.get("CLOUDFLARE_ACCOUNT_ID")
            or await ctx.env("CLOUDFLARE_ACCOUNT_ID")
        )
        if not token or not account:
            return None
        if self.kind == "ai-gateway":
            gateway = (
                options.get("cloudflare_gateway_id")
                or credential_env.get("CLOUDFLARE_GATEWAY_ID")
                or await ctx.env("CLOUDFLARE_GATEWAY_ID")
            )
            if not gateway:
                return None
            return AuthResult(
                auth={
                    "base_url": f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}",
                    "headers": {"cf-aig-authorization": f"Bearer {token}"},
                },
                env={"CLOUDFLARE_ACCOUNT_ID": account, "CLOUDFLARE_GATEWAY_ID": gateway},
            )
        return AuthResult(
            auth={
                "api_key": cast(str, token),
                "base_url": f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
            },
            env={"CLOUDFLARE_ACCOUNT_ID": account},
        )

    async def login(self, interaction: Any) -> dict[str, Any]:
        key = await interaction.prompt({"type": "secret", "message": "Enter Cloudflare API key"})
        account = await interaction.prompt(
            {"type": "text", "message": "Enter Cloudflare account ID"}
        )
        env = {"CLOUDFLARE_ACCOUNT_ID": account}
        if self.kind == "ai-gateway":
            gateway = await interaction.prompt(
                {"type": "text", "message": "Enter Cloudflare AI Gateway ID"}
            )
            env["CLOUDFLARE_GATEWAY_ID"] = gateway
        return {"type": "api_key", "key": key, "env": env}


def _cloudflare_base(kind: str) -> str:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not account:
        return ""
    if kind == "ai-gateway":
        gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID", "")
        return f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}"
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"


def _fetch_cloudflare_models(
    provider_id: str,
    kind: str,
    context: RefreshModelsContext,
) -> Any:
    base_url = _cloudflare_base(kind)
    if not base_url:
        return []
    return _fetch_openai_models(provider_id, base_url, "CLOUDFLARE_API_KEY", context)


def cloudflare_workers_ai_provider() -> Provider:
    return create_provider(
        id="cloudflare-workers-ai",
        name="Cloudflare Workers AI",
        auth=_CloudflareAuth("workers-ai"),  # type: ignore[arg-type]
        models=[],
        base_url="",
        api_kind="completions",
        fetch_models=lambda context: _fetch_cloudflare_models(
            "cloudflare-workers-ai", "workers-ai", context
        ),
    )


def cloudflare_ai_gateway_provider() -> Provider:
    return create_provider(
        id="cloudflare-ai-gateway",
        name="Cloudflare AI Gateway",
        auth=_CloudflareAuth("ai-gateway"),  # type: ignore[arg-type]
        models=[],
        base_url="",
        api_kind="completions",
        fetch_models=lambda context: _fetch_cloudflare_models(
            "cloudflare-ai-gateway", "ai-gateway", context
        ),
    )


__all__ = [
    "cloudflare_workers_ai_provider",
    "cloudflare_ai_gateway_provider",
]
