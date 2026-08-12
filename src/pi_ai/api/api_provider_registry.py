"""API 提供者注册表（对齐 TS compat.ts 的 apiProviderRegistry）。

把「API 协议 ID → stream / streamSimple 实现」统一注册、按需分发。
新增 API 协议（例如 google-generative-ai、mistral-conversations）只需
注册一次即可接入所有调度入口，不再改动 provider.py 的 if/elif 分发：

    from pi_ai import ApiProvider, register_api_provider

    register_api_provider(ApiProvider(
        api="my-api",
        stream=my_stream,            # (model, context, options) -> EventStream
        streamSimple=my_stream_simple,
    ))

之后 Provider.stream / 顶层 stream / stream_simple / complete /
complete_simple 自动按 model.api 分发。

内置注册（register_builtin_api_providers）：

    openai-completions → api/completions.py
    openai-responses   → api/responses.py
    pi-messages        → api/pi_messages_lazy.py

其中 completions / responses 的 streamSimple 暂复用 stream
（simple_options 仅移植了 max_tokens 上下文收敛 clamp_max_tokens_to_context，
其余 base options 字段仍由各 API 文件内联构建，见 api/simple_options.py）。
"""

import inspect

from dataclasses import dataclass
from typing import Awaitable, Callable, cast

from ..types import (
    AssistantMessage,
    Context,
    Model,
    ProviderEnv,
    SimpleStreamOptions,
    StreamOptions,
)
from ..utils._event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from .lazy import lazy_stream

# 注册表条目签名：返回 EventStream；async 实现返回 Awaitable（内部可 lazy），
# 与 TS ApiStreamFunction 对齐。
ApiStreamFunction = Callable[
    [Model, Context, StreamOptions | None],
    "Awaitable[AssistantMessageEventStream] | AssistantMessageEventStream",
]


@dataclass(slots=True)
class ApiProvider:
    """一个 API 协议实现（api ID + stream / streamSimple）。"""

    api: str
    stream: ApiStreamFunction
    streamSimple: ApiStreamFunction


@dataclass(slots=True)
class _RegisteredApiProvider:
    provider: ApiProvider
    source_id: str | None = None


_registry: dict[str, _RegisteredApiProvider] = {}

_BUILTIN_SOURCE_ID = "<builtin>"

# 顶层分发时 provider → 环境变量映射（对齐 TS env-api-keys.ts 的子集；
# 未知 provider 回退为 {PROVIDER}_API_KEY）。
_API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    # 本地服务（Ollama）无需 API Key。
    "ollama": "",
}

_AMBIENT_AUTH_MARKER = "<authenticated>"


def _wrap_stream(api: str, stream: ApiStreamFunction) -> ApiStreamFunction:
    """包装注册的 stream：model.api 不匹配时抛错（对齐 TS wrapStream）。"""

    def _wrapped(
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Awaitable[AssistantMessageEventStream] | AssistantMessageEventStream:
        if model.api and model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return stream(model, context, options)

    return _wrapped


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """注册 API 协议实现；同 api 的旧条目被覆盖。"""
    _registry[provider.api] = _RegisteredApiProvider(
        provider=ApiProvider(
            api=provider.api,
            stream=_wrap_stream(provider.api, provider.stream),
            streamSimple=_wrap_stream(provider.api, provider.streamSimple),
        ),
        source_id=source_id,
    )


def get_api_provider(api: str) -> ApiProvider | None:
    """按 API 协议 ID 取注册条目；未注册返回 None。"""
    entry = _registry.get(api)
    return entry.provider if entry is not None else None


def get_api_providers() -> list[ApiProvider]:
    """返回全部已注册 API 协议实现。"""
    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """按 source_id 注销（对齐 TS unregisterApiProviders）。"""
    for api, entry in list(_registry.items()):
        if entry.source_id == source_id:
            _registry.pop(api, None)


def _builtin_api_providers() -> list[ApiProvider]:
    """内置 API 实现（completions / responses / pi-messages）。"""
    from .completions import chat_completions_stream
    from .responses import responses_stream
    from .pi_messages_lazy import pi_messages_api
    from .google_generative_ai import google_stream, google_stream_simple
    from .mistral_conversations import mistral_stream, mistral_stream_simple
    from .azure_openai_responses import azure_stream, azure_stream_simple
    from .openai_codex_responses import codex_stream, codex_stream_simple
    from .google_vertex import vertex_stream, vertex_stream_simple
    from .bedrock_converse_stream import bedrock_stream, bedrock_stream_simple

    def _completions(
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Awaitable[AssistantMessageEventStream]:
        opts = options or {}
        return chat_completions_stream(
            model,
            context,
            opts.get("api_key", ""),
            opts.get("base_url") or model.base_url or "",
            options,
        )

    def _responses(
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> Awaitable[AssistantMessageEventStream]:
        opts = options or {}
        return responses_stream(
            model,
            context,
            opts.get("api_key", ""),
            opts.get("base_url") or model.base_url or "",
            options,
        )

    pi_messages = pi_messages_api()
    return [
        ApiProvider(
            api="openai-completions",
            stream=_completions,
            streamSimple=_completions,
        ),
        ApiProvider(
            api="openai-responses",
            stream=_responses,
            streamSimple=_responses,
        ),
        ApiProvider(
            api="pi-messages",
            stream=pi_messages.stream,
            streamSimple=pi_messages.streamSimple,
        ),
        ApiProvider(
            api="google-generative-ai",
            stream=google_stream,
            streamSimple=google_stream_simple,
        ),
        ApiProvider(
            api="mistral-conversations",
            stream=mistral_stream,
            streamSimple=mistral_stream_simple,
        ),
        ApiProvider(
            api="azure-openai-responses",
            stream=azure_stream,
            streamSimple=azure_stream_simple,
        ),
        ApiProvider(
            api="openai-codex-responses",
            stream=codex_stream,
            streamSimple=codex_stream_simple,
        ),
        ApiProvider(
            api="google-vertex",
            stream=vertex_stream,
            streamSimple=vertex_stream_simple,
        ),
        ApiProvider(
            api="bedrock-converse-stream",
            stream=bedrock_stream,
            streamSimple=bedrock_stream_simple,
        ),
    ]


def register_builtin_api_providers() -> None:
    """注册内置 API 实现；不覆盖已注册条目（对齐 TS registerBuiltInApiProviders）。"""
    for provider in _builtin_api_providers():
        if get_api_provider(provider.api) is None:
            register_api_provider(provider, source_id=_BUILTIN_SOURCE_ID)


def reset_api_providers() -> None:
    """清空注册表并重注册内置 API（对齐 TS resetApiProviders）。"""
    _registry.clear()
    register_builtin_api_providers()


# ---------------------------------------------------------------------------
# 顶层分发（对齐 TS compat.stream / streamSimple / complete / completeSimple）
# ---------------------------------------------------------------------------


def _get_env_api_key(provider: str, env: ProviderEnv | None = None) -> str | None:
    env_var = _API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
    if not env_var:
        return None
    value = get_provider_env_value(env_var, env)
    return value if value else None


def _with_env_api_key(
    model: Model,
    options: StreamOptions | None,
) -> StreamOptions | None:
    """未显式传 api_key 时按 provider 注入环境变量（对齐 TS withEnvApiKey）。"""
    if options and options.get("api_key"):
        return options
    api_key = _get_env_api_key(model.provider, options.get("env") if options else None)
    if not api_key or api_key == _AMBIENT_AUTH_MARKER:
        return options
    opts = dict(options or {})
    opts["api_key"] = api_key
    return cast(StreamOptions | None, opts)


def _resolve_api_provider(api: str) -> ApiProvider:
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def _invoke_sync(
    stream_fn: ApiStreamFunction,
    model: Model,
    context: Context,
    options: StreamOptions | None,
) -> AssistantMessageEventStream:
    """调用注册表条目；async 实现经 lazy_stream 包装后同步返回流。"""
    result = stream_fn(model, context, options)
    if inspect.isawaitable(result):
        return lazy_stream(model, lambda: result)  # type: ignore[arg-type]
    return result


async def invoke_api_stream(
    stream_fn: ApiStreamFunction,
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """async 上下文中的条目调用：同时兼容 async / sync 实现。"""
    result = stream_fn(model, context, options)
    if inspect.isawaitable(result):
        return await result
    return result


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """按 model.api 分发并同步返回事件流（env API key 注入）。"""
    entry = _resolve_api_provider(model.api)
    return _invoke_sync(entry.stream, model, context, _with_env_api_key(model, options))


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """streamSimple 分发（SimpleStreamOptions）。"""
    entry = _resolve_api_provider(model.api)
    return _invoke_sync(
        entry.streamSimple,
        model,
        context,
        _with_env_api_key(model, options),
    )


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """非流式分发：等待流结束返回最终 AssistantMessage。"""
    event_stream = stream(model, context, options)
    return await event_stream.result()


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """completeSimple 分发。"""
    event_stream = stream_simple(model, context, options)
    return await event_stream.result()


register_builtin_api_providers()


__all__ = [
    "ApiProvider",
    "ApiStreamFunction",
    "register_api_provider",
    "get_api_provider",
    "get_api_providers",
    "unregister_api_providers",
    "register_builtin_api_providers",
    "reset_api_providers",
    "invoke_api_stream",
    "stream",
    "stream_simple",
    "complete",
    "complete_simple",
]
