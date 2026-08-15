"""内置图片 API 提供者注册。"""

from ..images_api_registry import register_images_api_provider
from ..images_models import create_images_models
from .openrouter_images import (
    generate_images as _generate_openrouter_images,
    openrouter_images_provider,
)

_registered = False


def register_builtin_images_api_providers() -> None:
    """注册内置图片 API（幂等）。"""
    global _registered
    if _registered:
        return
    register_images_api_provider(
        "openrouter-images",
        _generate_openrouter_images,
        source_id="builtin",
    )
    _registered = True


def builtin_images_providers():
    """全部内置图片 provider（当前为 OpenRouter）。"""
    return [openrouter_images_provider()]


def builtin_images_models():
    """包含全部内置图片 provider 的 ImagesModels 实例（对齐 TS）。"""
    models = create_images_models()
    for provider in builtin_images_providers():
        models.set_provider(provider)
    return models


__all__ = [
    "register_builtin_images_api_providers",
    "builtin_images_providers",
    "builtin_images_models",
]
