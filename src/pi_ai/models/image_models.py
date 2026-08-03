"""图片模型目录（对齐 TS image-models.ts / image-models.generated.ts）。

当前唯一的图片 provider 是 OpenRouter；目录为静态数据，
后续可并入 src/pi_ai/scripts/generate_models.py 的生成体系。
"""

from .._types import ImagesModel

OPENROUTER_IMAGE_MODELS: list[ImagesModel] = [
    ImagesModel(
        id="openai/gpt-image-1",
        api="openrouter-images",
        provider="openrouter",
        name="OpenAI GPT Image 1",
        input=["text", "image"],
        output=["text", "image"],
    ),
    ImagesModel(
        id="google/gemini-2.5-flash-image-preview",
        api="openrouter-images",
        provider="openrouter",
        name="Gemini 2.5 Flash Image Preview",
        input=["text", "image"],
        output=["text", "image"],
    ),
    ImagesModel(
        id="black-forest-labs/flux-1.1-pro",
        api="openrouter-images",
        provider="openrouter",
        name="FLUX 1.1 Pro",
        input=["text", "image"],
        output=["image"],
    ),
    ImagesModel(
        id="ideogram/ideogram-v2-turbo",
        api="openrouter-images",
        provider="openrouter",
        name="Ideogram V2 Turbo",
        input=["text"],
        output=["image"],
    ),
    ImagesModel(
        id="recraft/recraft-v3",
        api="openrouter-images",
        provider="openrouter",
        name="Recraft V3",
        input=["text"],
        output=["image"],
    ),
]


def get_image_model(provider: str, model_id: str) -> ImagesModel | None:
    return next(
        (m for m in get_image_models(provider) if m.id == model_id), None
    )


def get_image_providers() -> list[str]:
    return ["openrouter"]


def get_image_models(provider: str) -> list[ImagesModel]:
    if provider == "openrouter":
        return list(OPENROUTER_IMAGE_MODELS)
    return []


__all__ = [
    "OPENROUTER_IMAGE_MODELS",
    "get_image_model",
    "get_image_providers",
    "get_image_models",
]
