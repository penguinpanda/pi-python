"""OpenRouter 图片生成实现测试（mock HTTP）。"""

import asyncio

import httpx
import pytest

from pi_ai.providers import openrouter_images
from pi_ai.types import ImagesContext, ImagesModel


def _image_model() -> ImagesModel:
    return ImagesModel(
        id="openai/gpt-image-1",
        api="openrouter-images",
        provider="openrouter",
        input=["text", "image"],
        output=["text", "image"],
    )


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_build_params_text_and_image():
    model = _image_model()
    context: ImagesContext = {
        "input": [
            {"type": "text", "text": "make it blue"},
            {"type": "image", "mime_type": "image/png", "data": "aGk=", "url": None},
        ]
    }
    params = openrouter_images._build_params(model, context)
    assert params["stream"] is False
    assert params["modalities"] == ["image", "text"]
    content = params["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aGk="


@pytest.mark.asyncio
async def test_generate_images_success(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "choices": [
                    {
                        "message": {
                            "content": "Here is your image",
                            "images": [
                                {
                                    "image_url": {
                                        "url": "data:image/png;base64,aGVsbG8="
                                    }
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        )

    monkeypatch.setattr(
        openrouter_images,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = await openrouter_images.generate_images(
        _image_model(),
        {"input": [{"type": "text", "text": "cat"}]},
        {"api_key": "sk-or"},
    )
    assert result["stop_reason"] == "stop"
    assert result["response_id"] == "gen-1"
    assert result["output"][0]["type"] == "text"
    assert result["output"][1] == {
        "type": "image",
        "mime_type": "image/png",
        "data": "aGVsbG8=",
    }
    assert result["usage"]["output"] == 20
    assert result["usage"]["cache_read"] == 4


@pytest.mark.asyncio
async def test_generate_images_http_error_returns_error_output(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(
        openrouter_images,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    result = await openrouter_images.generate_images(
        _image_model(),
        {"input": [{"type": "text", "text": "cat"}]},
        {"api_key": "sk-or"},
    )
    assert result["stop_reason"] == "error"
    assert "error_message" in result


@pytest.mark.asyncio
async def test_generate_images_missing_key():
    result = await openrouter_images.generate_images(
        _image_model(), {"input": []}, {}
    )
    assert result["stop_reason"] == "error"
    assert "No API key" in result["error_message"]


@pytest.mark.asyncio
async def test_generate_images_abort_via_signal(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    monkeypatch.setattr(
        openrouter_images,
        "_AsyncClient",
        lambda **kwargs: _mock_client(handler),
    )
    signal = asyncio.Event()
    signal.set()
    result = await openrouter_images.generate_images(
        _image_model(),
        {"input": []},
        {"api_key": "sk-or", "signal": signal},
    )
    assert result["stop_reason"] == "aborted"
