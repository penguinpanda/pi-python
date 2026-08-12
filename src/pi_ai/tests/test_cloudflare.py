"""Cloudflare provider 认证与配置测试。"""

from __future__ import annotations

from pi_ai.auth.context import AuthContext
from pi_ai import Models
from pi_ai.providers.cloudflare import (
    _CloudflareAuth,
    cloudflare_ai_gateway_provider,
    cloudflare_workers_ai_provider,
)


class _EnvContext(AuthContext):
    def __init__(self, env: dict[str, str]) -> None:
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


async def test_cloudflare_workers_auth() -> None:
    auth = _CloudflareAuth("workers-ai")
    result = await auth.resolve_auth(
        None,  # type: ignore[arg-type]
        _EnvContext({"CLOUDFLARE_API_KEY": "token", "CLOUDFLARE_ACCOUNT_ID": "acct"}),
        {},
    )
    assert result is not None
    assert result.auth["base_url"] == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"
    assert result.auth["api_key"] == "token"


async def test_cloudflare_gateway_auth() -> None:
    auth = _CloudflareAuth("ai-gateway")
    result = await auth.resolve_auth(
        None,  # type: ignore[arg-type]
        _EnvContext(
            {
                "CLOUDFLARE_API_KEY": "token",
                "CLOUDFLARE_ACCOUNT_ID": "acct",
                "CLOUDFLARE_GATEWAY_ID": "gw",
            }
        ),
        {},
    )
    assert result is not None
    assert result.auth["base_url"] == "https://gateway.ai.cloudflare.com/v1/acct/gw"
    assert result.auth["headers"]["cf-aig-authorization"] == "Bearer token"


def test_cloudflare_providers_registered() -> None:
    assert cloudflare_workers_ai_provider().id == "cloudflare-workers-ai"
    assert cloudflare_ai_gateway_provider().id == "cloudflare-ai-gateway"


async def test_cloudflare_check_auth() -> None:
    models = Models(
        auth_context=_EnvContext(
            {
                "CLOUDFLARE_API_KEY": "token",
                "CLOUDFLARE_ACCOUNT_ID": "acct",
                "CLOUDFLARE_GATEWAY_ID": "gw",
            }
        )
    )
    models.add_provider(cloudflare_workers_ai_provider())
    assert await models.check_auth("cloudflare-workers-ai") == {
        "type": "api_key",
        "source": "custom",
    }


class _FakeInteraction:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers

    async def prompt(self, prompt) -> str:  # type: ignore[no-untyped-def]
        return self.answers.pop(0)

    def notify(self, event) -> None:  # type: ignore[no-untyped-def]
        pass


async def test_cloudflare_login() -> None:
    models = Models()
    models.add_provider(cloudflare_workers_ai_provider())
    credential = await models.login(
        "cloudflare-workers-ai",
        _FakeInteraction(["token", "acct"]),
    )
    assert credential["key"] == "token"
    assert credential["env"]["CLOUDFLARE_ACCOUNT_ID"] == "acct"
