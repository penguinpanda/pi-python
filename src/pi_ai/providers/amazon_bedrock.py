"""AWS Bedrock provider（bearer token 认证）。"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from pi_ai.auth import ResolvedAuth
from pi_ai.auth.types import AuthContext, AuthResult, CredentialStore
from pi_ai.provider import Provider, create_provider
from pi_ai.types import Model


def _ambient_aws_credential_source(env: dict[str, str]) -> str | None:
    """检测无需 bearer token 的 ambient AWS 配置来源（对齐 TS bedrockAuth.resolve）。"""
    if env.get("AWS_PROFILE") or env.get("AWS_DEFAULT_PROFILE"):
        return "AWS_PROFILE"
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        return "AWS access keys"
    for key in (
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        if env.get(key):
            return key
    return None


def _ambient_env_from_os() -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in (
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
        if os.environ.get(key)
    }
    return env


def _shared_credentials_exists(env: dict[str, str]) -> bool:
    raw = env.get("AWS_SHARED_CREDENTIALS_FILE") or str(Path.home() / ".aws" / "credentials")
    try:
        return Path(raw).expanduser().is_file()
    except OSError:
        return False


class _BedrockAuth:
    """Bearer token 或 AWS 默认凭证链。

    bearer token 返回 api_key；ambient AWS 配置返回空 api_key 的
    ``AuthResult(auth={})``，由 bedrock API 层执行 SigV4 或读取
    shared credentials file。
    """

    display_name = "AWS credentials or bearer token"
    env_vars = ["AWS_BEARER_TOKEN_BEDROCK"]

    def resolve(self, credential=None) -> ResolvedAuth | None:  # type: ignore[no-untyped-def]
        env: dict[str, str] = {}
        if isinstance(credential, dict):
            env.update({k: v for k, v in credential.get("env", {}).items() if isinstance(v, str)})
        key = None
        if isinstance(credential, dict):
            key = credential.get("key")
        else:
            key = getattr(credential, "key", None)
        env.update(_ambient_env_from_os())
        token = key or env.get("AWS_BEARER_TOKEN_BEDROCK")
        if token:
            return ResolvedAuth(
                api_key=str(token),
                source="stored credential" if key else "AWS_BEARER_TOKEN_BEDROCK",
            )
        source = _ambient_aws_credential_source(env)
        if source is None and _shared_credentials_exists(env):
            source = "~/.aws/credentials"
        if source is not None:
            return ResolvedAuth(api_key="", source=source)
        return None

    async def resolve_auth(
        self,
        store: CredentialStore,
        ctx: AuthContext,
        options: dict[str, Any],
    ) -> AuthResult | None:
        credential: Any = None
        try:
            credential = await store.read("amazon-bedrock")
        except Exception:
            credential = None

        async def env(name: str) -> str | None:
            value = (options.get("env") or {}).get(name)
            if value is None and isinstance(credential, dict):
                value = credential.get("env", {}).get(name)
            if value is None:
                value = await ctx.env(name)
            return value or None

        credential_key = (
            credential.get("key")
            if isinstance(credential, dict)
            else getattr(credential, "key", None)
        )
        token = options.get("api_key") or credential_key or await env("AWS_BEARER_TOKEN_BEDROCK")
        if token:
            return AuthResult(
                auth={"api_key": str(token)},
                source="stored credential" if credential_key else "AWS_BEARER_TOKEN_BEDROCK",
            )

        resolved_env: dict[str, str] = {}
        for name in (
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        ):
            value = await env(name)
            if value:
                resolved_env[name] = value

        source = _ambient_aws_credential_source(resolved_env)
        if source is None and await ctx.file_exists(
            resolved_env.get("AWS_SHARED_CREDENTIALS_FILE") or "~/.aws/credentials"
        ):
            source = "~/.aws/credentials"
        if source is None:
            return None
        return AuthResult(auth={}, env=resolved_env or None, source=source)


BEDROCK_MODELS: list[Model] = [
    Model(
        id="anthropic.claude-sonnet-4-20250514",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Claude Sonnet 4 on Bedrock",
        input=["text", "image"],
        output=["text"],
        max_tokens=32000,
        context_window=200000,
        reasoning=True,
    ),
    Model(
        id="amazon.nova-pro-v1:0",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Amazon Nova Pro",
        input=["text", "image"],
        output=["text"],
        max_tokens=32768,
        context_window=300000,
    ),
    Model(
        id="meta.llama3-1-405b-instruct-v1:0",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        name="Llama 3.1 405B on Bedrock",
        input=["text"],
        output=["text"],
        max_tokens=32768,
        context_window=131072,
    ),
]


def amazon_bedrock_provider() -> Provider:
    return create_provider(
        id="amazon-bedrock",
        name="AWS Bedrock",
        auth=_BedrockAuth(),  # type: ignore[arg-type]
        models=BEDROCK_MODELS,
        base_url="",
        api_kind="bedrock-converse-stream",
    )


__all__ = ["BEDROCK_MODELS", "amazon_bedrock_provider"]
