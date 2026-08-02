"""图片 API 提供者注册表（对齐 TS images-api-registry.ts）。"""

from typing import Any, Awaitable, Callable

from ._types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions

ImagesApiFunction = Callable[
    [ImagesModel, ImagesContext, ImagesOptions | None],
    Awaitable[AssistantImages],
]

_registry: dict[str, ImagesApiFunction] = {}
_sources: dict[str, str] = {}


def register_images_api_provider(
    api: str,
    generate_images: ImagesApiFunction,
    source_id: str | None = None,
) -> None:
    """注册图片 API 实现；api 与 model.api 不匹配时抛错。"""

    async def _wrapped(
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return await generate_images(model, context, options)

    _registry[api] = _wrapped
    if source_id is not None:
        _sources[api] = source_id


def get_images_api_provider(api: str) -> ImagesApiFunction | None:
    return _registry.get(api)


def unregister_images_api_providers(source_id: str) -> None:
    for api, source in list(_sources.items()):
        if source == source_id:
            _registry.pop(api, None)
            _sources.pop(api, None)


def clear_images_api_providers() -> None:
    _registry.clear()
    _sources.clear()


__all__ = [
    "ImagesApiFunction",
    "register_images_api_provider",
    "get_images_api_provider",
    "unregister_images_api_providers",
    "clear_images_api_providers",
]
