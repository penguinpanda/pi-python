"""图片模型目录测试。"""

from pi_ai.models.image_models import (
    OPENROUTER_IMAGE_MODELS,
    get_image_model,
    get_image_models,
    get_image_providers,
)


def test_image_catalog_basic():
    assert get_image_providers() == ["openrouter"]
    assert len(get_image_models("openrouter")) == len(OPENROUTER_IMAGE_MODELS)
    assert get_image_models("unknown") == []


def test_get_image_model():
    model = get_image_model("openrouter", "openai/gpt-image-1")
    assert model is not None
    assert model.api == "openrouter-images"
    assert model.provider == "openrouter"
    assert get_image_model("openrouter", "nope") is None
