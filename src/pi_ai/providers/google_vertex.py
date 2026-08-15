"""Google Vertex AI provider。"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from pi_ai.auth import ResolvedAuth
from pi_ai.auth.types import AuthContext, AuthResult, CredentialStore
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model

VERTEX_ADC_PATH = "~/.config/gcloud/application_default_credentials.json"


def _credential_env(credential: Any) -> dict[str, str]:
    if isinstance(credential, dict):
        return {k: v for k, v in credential.get("env", {}).items() if isinstance(v, str)}
    return {}


def _has_adc(env: dict[str, str]) -> bool:
    path = env.get("GOOGLE_APPLICATION_CREDENTIALS") or VERTEX_ADC_PATH
    try:
        return Path(path).expanduser().is_file()
    except OSError:
        return False


class _GoogleVertexAuth:
    """Vertex 支持 bearer token、Cloud API key 或 Application Default Credentials。

    ambient（ADC / API key）返回空 api_key 的 ``AuthResult(auth={})``，
    API 层自行刷新 ADC token 或设置 x-goog-api-key。
    """

    display_name = "Google Cloud credentials"
    env_vars = ["GOOGLE_OAUTH_ACCESS_TOKEN"]

    def resolve(self, credential=None) -> ResolvedAuth | None:  # type: ignore[no-untyped-def]
        env = _credential_env(credential)
        for name in (
            "GOOGLE_CLOUD_API_KEY",
            "GOOGLE_OAUTH_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GCP_PROJECT",
            "GCLOUD_PROJECT",
        ):
            if os.environ.get(name):
                env.setdefault(name, os.environ[name])
        key = None
        if isinstance(credential, dict):
            key = credential.get("key")
        else:
            key = getattr(credential, "key", None)
        if key:
            return ResolvedAuth(api_key=str(key), source="stored credential")
        if env.get("GOOGLE_CLOUD_API_KEY"):
            return ResolvedAuth(api_key=env["GOOGLE_CLOUD_API_KEY"], source="GOOGLE_CLOUD_API_KEY")
        if env.get("GOOGLE_OAUTH_ACCESS_TOKEN"):
            return ResolvedAuth(
                api_key=env["GOOGLE_OAUTH_ACCESS_TOKEN"], source="GOOGLE_OAUTH_ACCESS_TOKEN"
            )
        if _has_adc(env):
            return ResolvedAuth(api_key="", source="gcloud application default credentials")
        return None

    async def resolve_auth(
        self,
        store: CredentialStore,
        ctx: AuthContext,
        options: dict[str, Any],
    ) -> AuthResult | None:
        credential: Any = None
        try:
            credential = await store.read("google-vertex")
        except Exception:
            credential = None

        async def env(name: str) -> str | None:
            value = (options.get("env") or {}).get(name)
            if value is None:
                value = _credential_env(credential).get(name)
            if value is None:
                value = await ctx.env(name)
            return value or None

        credential_key = (
            credential.get("key")
            if isinstance(credential, dict)
            else getattr(credential, "key", None)
        )
        explicit_key = options.get("api_key") or credential_key
        if explicit_key:
            cloud_api_key = await env("GOOGLE_CLOUD_API_KEY")
            kind = "api_key" if cloud_api_key and explicit_key == cloud_api_key else "token"
            return AuthResult(
                auth={"api_key": str(explicit_key)},
                env={"GOOGLE_CLOUD_API_KEY": str(explicit_key)} if kind == "api_key" else None,
                source="stored credential"
                if credential_key
                else ("GOOGLE_CLOUD_API_KEY" if kind == "api_key" else "GOOGLE_OAUTH_ACCESS_TOKEN"),
            )

        resolved_env: dict[str, str] = {}
        for name in (
            "GOOGLE_CLOUD_API_KEY",
            "GOOGLE_OAUTH_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GCP_PROJECT",
            "GCLOUD_PROJECT",
        ):
            value = await env(name)
            if value:
                resolved_env[name] = value

        if resolved_env.get("GOOGLE_CLOUD_API_KEY"):
            return AuthResult(
                auth={"api_key": resolved_env["GOOGLE_CLOUD_API_KEY"]},
                env=resolved_env or None,
                source="GOOGLE_CLOUD_API_KEY",
            )
        if resolved_env.get("GOOGLE_OAUTH_ACCESS_TOKEN"):
            return AuthResult(
                auth={"api_key": resolved_env["GOOGLE_OAUTH_ACCESS_TOKEN"]},
                env=resolved_env or None,
                source="GOOGLE_OAUTH_ACCESS_TOKEN",
            )
        has_adc = _has_adc(resolved_env)
        if has_adc is False:
            has_adc = await ctx.file_exists(
                resolved_env.get("GOOGLE_APPLICATION_CREDENTIALS") or VERTEX_ADC_PATH
            )
        if has_adc:
            return AuthResult(
                auth={},
                env=resolved_env or None,
                source="gcloud application default credentials",
            )
        return None


GOOGLE_VERTEX_MODELS: list[Model] = [
    Model(
        id="gemini-2.5-pro",
        provider="google-vertex",
        api="google-vertex",
        name="Vertex Gemini 2.5 Pro",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
    Model(
        id="gemini-2.5-flash",
        provider="google-vertex",
        api="google-vertex",
        name="Vertex Gemini 2.5 Flash",
        input=["text", "image"],
        output=["text"],
        max_tokens=65536,
        context_window=1000000,
        reasoning=True,
    ),
]


def google_vertex_provider() -> Provider:
    return create_provider(
        id="google-vertex",
        name="Google Vertex",
        auth=_GoogleVertexAuth(),  # type: ignore[arg-type]
        models=GOOGLE_VERTEX_MODELS,
        base_url="",
        api_kind="google-vertex",
    )


__all__ = ["GOOGLE_VERTEX_MODELS", "google_vertex_provider"]
