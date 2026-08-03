"""图片生成顶级入口（对齐 TS images.ts）。"""

from .types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions
from .images_api_registry import get_images_api_provider
from .providers.images import register_builtin_images_api_providers

# 注册内置图片 API 提供者（openrouter-images）。
register_builtin_images_api_providers()


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions | None = None,
) -> AssistantImages:
    """按 model.api 查找注册表并生成图片。"""
    provider = get_images_api_provider(model.api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {model.api}")
    return await provider(model, context, options)


__all__ = ["generate_images"]
