"""pi_ai.types.image — 图片生成类型（类型先行，无运行时实现）。"""

import asyncio

from dataclasses import dataclass, field

from typing import Any, Callable, Literal, TypedDict

from typing_extensions import NotRequired

from .common import (
    AsyncHTTPClient,
    ProviderEnv,
    ProviderHeaders,
)
from .content import ImageContent, TextContent
from .message import Usage


KnownImagesApi = Literal["openrouter-images"]
KnownImagesProvider = Literal["openrouter"]

ImagesApi = KnownImagesApi | str
ImagesProviderId = KnownImagesProvider | str

ImagesStopReason = Literal["stop", "error", "aborted"]

ImagesInputContent = TextContent | ImageContent
ImagesOutputContent = TextContent | ImageContent


class ImagesContext(TypedDict):
    input: list[ImagesInputContent]


class AssistantImages(TypedDict):
    api: ImagesApi
    provider: ImagesProviderId
    model: str
    output: list[ImagesOutputContent]
    response_id: NotRequired[str]
    usage: NotRequired[Usage]
    stop_reason: ImagesStopReason
    error_message: NotRequired[str]
    timestamp: int


@dataclass(slots=True)
class ImagesModel:
    id: str
    api: ImagesApi
    provider: ImagesProviderId
    name: str = ""
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    headers: dict[str, str] | None = None


class ImagesOptions(TypedDict, total=False):
    """图片生成请求参数"""

    signal: NotRequired[asyncio.Event]
    api_key: str
    http_client: AsyncHTTPClient
    env: ProviderEnv
    on_payload: Callable[..., Any]
    on_response: Callable[..., Any]
    headers: ProviderHeaders
    timeout_ms: int
    max_retries: int
    max_retry_delay_ms: int
    metadata: dict[str, Any]


ProviderImagesOptions = ImagesOptions


__all__ = [
    "KnownImagesApi",
    "KnownImagesProvider",
    "ImagesApi",
    "ImagesProviderId",
    "ImagesStopReason",
    "ImagesInputContent",
    "ImagesOutputContent",
    "ImagesContext",
    "AssistantImages",
    "ImagesModel",
    "ImagesOptions",
    "ProviderImagesOptions",
]
