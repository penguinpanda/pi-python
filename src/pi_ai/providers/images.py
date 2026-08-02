"""内置图片 API 提供者注册。"""

from ..images_api_registry import register_images_api_provider
from .openrouter_images import generate_images as _generate_openrouter_images

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


__all__ = ["register_builtin_images_api_providers"]
