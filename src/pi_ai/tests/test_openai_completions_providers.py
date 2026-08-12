"""OpenAI 兼容 completions provider 动态发现测试。"""

from __future__ import annotations

import httpx

from pi_ai.auth import ApiKeyCredential
from pi_ai.provider import RefreshModelsContext
from pi_ai.providers.openai_completions_providers import (
    _fetch_openai_models,
    cerebras_provider,
    groq_provider,
)


def test_provider_config() -> None:
    groq = groq_provider()
    assert groq.id == "groq"
    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert groq.auth is not None and groq.auth.env_vars == ["GROQ_API_KEY"]
    assert groq.refresh_models is not None

    cerebras = cerebras_provider()
    assert cerebras.id == "cerebras"
    assert cerebras.base_url == "https://api.cerebras.ai/v1"


async def test_fetch_openai_models(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})

    monkeypatch.setattr(
        "pi_ai.providers.openai_completions_providers._AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    context = RefreshModelsContext(
        credential=ApiKeyCredential(key="sk-test"),
        store=None,
        allow_network=True,
        force=False,
        signal=None,
    )
    models = await _fetch_openai_models(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", context
    )
    assert [model.id for model in models] == ["model-a", "model-b"]
    assert all(model.api == "openai-completions" for model in models)
