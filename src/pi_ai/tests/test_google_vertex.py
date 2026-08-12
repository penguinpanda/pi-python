"""Google Vertex provider/API 测试。"""

from __future__ import annotations

from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.google_vertex import _resolve_vertex_options
from pi_ai._types import Model
from pi_ai.providers.google_vertex import google_vertex_provider


def _model() -> Model:
    return Model(id="gemini-2.5-flash", provider="google-vertex", api="google-vertex")


def test_google_vertex_api_registered() -> None:
    assert get_api_provider("google-vertex") is not None


def test_google_vertex_provider_env_auth() -> None:
    provider = google_vertex_provider()
    assert provider.auth is not None and provider.auth.env_vars == ["GOOGLE_OAUTH_ACCESS_TOKEN"]
    assert provider.get_models()


def test_resolve_vertex_options_builds_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
    token, endpoint = _resolve_vertex_options(
        _model(),
        "token",
        "",
        {"env": {}},
    )
    assert token == "token"
    assert endpoint == (
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/asia-east1/publishers/google"
    )
