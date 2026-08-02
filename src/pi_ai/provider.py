"""
Provider（模型提供者）

=========================================================
模块职责
=========================================================

Provider 表示一个已经配置好的模型服务提供者。

例如：

    OpenAI

    DeepSeek

    Azure OpenAI（以后可扩展）

一个 Provider 负责：

    ① 保存 Provider 配置

        - 名称
        - Base URL
        - API 类型

    ② 保存支持的模型列表

    ③ 负责认证（API Key）

    ④ 将请求调度到具体 API 实现

Provider 本身不负责：

    - HTTP 请求实现
    - EventStream
    - 消息格式转换

这些工作分别由：

    api/
        completions.py
        responses.py

负责。


=========================================================
整体关系
=========================================================

             Models
                │
                ▼
          Provider(OpenAI)
                │
         resolve_api_key()
                │
                ▼
      chat_completions_stream()

或者

      responses_stream()
"""

import asyncio

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol

from .utils._event_stream import AssistantMessageEventStream
from ._types import (
    AssistantMessage,
    AsyncHTTPClient,
    Context,
    Model,
    StreamFunction,
    StreamOptions,
)

from .api.completions import chat_completions_stream
from .api.responses import responses_stream
from .auth import EnvApiKeyAuth, InMemoryCredentialStore, resolve_api_key
from .models.models_store import (
    ModelsStoreEntry,
    ProviderModelsStore,
    provider_models_store,
)
from ._types import now_ms

# Provider 使用的 API 类型。
#
# completions
#     Chat Completions API
#
# responses
#     Responses API

ApiKind = Literal["completions", "responses"]


@dataclass(slots=True)
class RefreshModelsContext:
    """provider 侧 refreshModels 的上下文（对齐 TS RefreshModelsContext）。"""

    # 生效中的凭证（OAuth 已按需刷新）；API Key provider 可忽略。
    credential: Any = None
    # 当前 provider 的持久化存储视图（只读自己的目录）。
    store: ProviderModelsStore | None = None
    # False 表示离线/仅缓存初始化。
    allow_network: bool = True
    # 绕过新鲜度检查立即抓取。
    force: bool = False
    # 可选中止信号（asyncio.Event；set 即中止）。
    signal: asyncio.Event | None = None

# 自定义流函数类型（定义已上移到 _types.py，此处从 _types 导入）。
#
# 当 Provider 设置了 _stream_fn 时，
# stream() 会直接调用它，
# 跳过：
#
#     - API Key 解析
#     - API 类型分发（completions / responses）
#
# 通常用于测试（例如 Faux Provider）。
@dataclass(slots=True)
class Provider:
    """
    模型提供者。

    一个 Provider 表示：

        一个模型服务商
        +
        一套认证方式
        +
        一组模型
        +
        一个 API 实现

    例如：

        OpenAI

            auth
            models
            completions API

        DeepSeek

            auth
            models
            responses API（未来可支持）
    """

    # Provider 唯一 ID
    #
    # 例如：
    #
    # "openai"
    # "deepseek"
    id: str

    # 显示名称
    #
    # 用于日志、CLI 等。
    name: str

    # API Key 认证策略
    #
    # Provider 获取 API Key 时，
    # 会通过这里解析。
    #
    # auth=None 表示不需要认证，
    # 例如本地服务（Ollama）。
    auth: EnvApiKeyAuth | None

    # Provider 支持的模型列表。
    models: list[Model]

    # 底层 API 类型。
    #
    # 决定调用：
    #
    # completions API
    #
    # 或
    #
    # responses API
    _api_kind: ApiKind = "completions"

    # Provider Base URL。
    #
    # 默认使用官方地址。
    #
    # 可以用于：
    #
    # Azure
    # OpenRouter
    # Ollama
    base_url: str | None = None

    # Provider 使用的 Credential Store。
    #
    # 默认使用内存实现。
    _credential_store: InMemoryCredentialStore = field(default_factory=InMemoryCredentialStore)

    # 自定义流函数（可选）。
    #
    # 设置后，stream() 会直接调用它，
    # 跳过 API Key 解析与 API 类型分发。
    #
    # 主要用于测试（例如 Faux Provider）。
    _stream_fn: StreamFunction | None = None

    # 运行时动态发现的模型（fetch_models 抓取结果，覆盖同 id 的静态模型）。
    _dynamic_models: list[Model] = field(default_factory=list, repr=False, compare=False)

    # 动态模型刷新实现（由 create_provider(fetch_models=...) 构建）。
    refresh_models: Callable[[RefreshModelsContext], Awaitable[None]] | None = None

    # 自定义异步 HTTP 客户端（可选）。
    #
    # 设置后用于 Provider 的 HTTP 请求；
    # None 时使用默认客户端（openai SDK / httpx）。
    http_client: AsyncHTTPClient | None = None

    def get_models(self) -> list[Model]:
        """
        返回 Provider 支持的所有模型。

        返回副本，

        避免调用者修改内部列表。
        """

        if not self._dynamic_models:
            return list(self.models)
        merged = list(self.models)
        for model in self._dynamic_models:
            for index, existing in enumerate(merged):
                if existing.id == model.id:
                    merged[index] = model
                    break
            else:
                merged.append(model)
        return merged

    async def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """
        发送流式聊天请求。

        工作流程：

                Provider

                    │

                    ▼

            resolve_api_key()

                    │

                    ▼

            根据 API 类型

                    │

            ┌──────┴──────┐

            ▼             ▼

        completions    responses

                    │

                    ▼

        AssistantMessageEventStream
        """

        # 自定义流函数。
        #
        # 如果设置了 _stream_fn（例如 Faux Provider），
        # 直接调用它，跳过：
        #
        #     - API Key 解析
        #     - API 类型分发（completions / responses）
        #
        # 这样 Faux Provider 完全不需要认证与网络。
        if self._stream_fn is not None:
            return await self._stream_fn(model, context, options)

        # 解析 API Key。
        #
        # 优先级：
        #
        # StreamOptions.api_key（本次请求覆盖）
        #
        #       ↓
        #
        # Credential Store
        #
        #       ↓
        #
        # Environment Variable
        #
        # auth=None（本地服务，如 Ollama）：
        # 跳过认证解析，使用占位值。
        # OpenAI SDK 在发请求时要求 api_key 非空，
        # 本地服务会忽略 Authorization 头。
        opts = options or {}
        api_key = opts.get("api_key")
        if api_key is None:
            if self.auth is None:
                api_key = "ollama"
            else:
                api_key = await resolve_api_key(self.auth, self._credential_store, self.id)
        base_url = self.base_url or ""

        # 根据 Provider 使用的 API 类型，
        # 调度到不同实现。
        #
        # 不同 Provider 可以共享同一套 Models，
        # 但底层 API 实现不同。

        # responses：适合构建复杂的 AI Agent，例如需要联网搜索、文件检索、自动执行代码或多步推理的智能应用。
        if self._api_kind == "responses":
            return await responses_stream(model, context, api_key, base_url, options)

        # completions：适合简单的文本生成任务，如基础聊天机器人、内容总结、分类等。
        elif self._api_kind == "completions":
            return await chat_completions_stream(model, context, api_key, base_url, options)

        # 未知 API 类型。
        #
        # 正常情况不会发生（_api_kind 是字面量类型），
        # 防御性处理，避免隐式返回 None。
        else:
            raise ValueError(f"Unknown API kind: {self._api_kind}")

    async def complete(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
    ) -> AssistantMessage:
        """
        非流式调用。

        内部仍然通过 stream() 实现。

        区别仅在于：

        stream()

        返回：

            EventStream

        complete()

        等待整个流结束，

        直接返回最终 AssistantMessage。
        """

        stream = await self.stream(model, context, options)
        return await stream.result()

def create_provider(
        id: str,
        name: str,
        auth: EnvApiKeyAuth | None,
        models: list[Model],
        api_kind: ApiKind = "completions",
        base_url: str | None = None,
        stream_fn: StreamFunction | None = None,
        fetch_models: (
            Callable[[RefreshModelsContext], Awaitable[list[Model]]] | None
        ) = None,
) -> Provider:
    """
    创建 Provider。

    相比直接调用 Provider(...)

    提供：

    - 默认值
    - 更简单的参数

    auth 传 None：

        表示 Provider 不需要 API Key，
        例如本地服务（Ollama）。

    stream_fn：

        可选的自定义流函数。

        设置后，stream() 会直接调用它，
        跳过认证与 API 类型分发。

        主要用于测试（例如 Faux Provider）。

    主要用于：

    Provider 注册。
    """

    provider = Provider(
        id=id,
        name=name or id,
        auth=auth,
        models=models,
        _api_kind=api_kind,
        base_url=base_url,
        _stream_fn=stream_fn,
    )

    if fetch_models is not None:
        provider.refresh_models = _build_refresh_models(provider, fetch_models)
    return provider


def _build_refresh_models(
    provider: Provider,
    fetch_models: Callable[[RefreshModelsContext], Awaitable[list[Model]]],
) -> Callable[[RefreshModelsContext], Awaitable[None]]:
    """构建 provider.refresh_models：缓存恢复 → 网络抓取 → 持久化。

    对齐 TS createProvider 的 refreshModels：
    - 先恢复 store 中的目录（allow_network=False 时仅此一步）；
    - 网络失败时保留上一次的列表（dynamic 不变），错误由 Models.refresh 收集；
    - 并发调用共享同一个 in-flight 任务。
    """
    inflight: asyncio.Task | None = None

    async def _refresh(context: RefreshModelsContext) -> None:
        nonlocal inflight
        if inflight is not None:
            await inflight
            return

        async def _impl() -> None:
            if context.store is not None:
                stored = await context.store.read()
                if stored is not None:
                    provider._dynamic_models[:] = [
                        m for m in stored.models if m.provider == provider.id
                    ]
            if not context.allow_network or (
                context.signal is not None and context.signal.is_set()
            ):
                return
            refreshed = await fetch_models(context)
            if context.signal is not None and context.signal.is_set():
                return
            provider._dynamic_models[:] = refreshed
            if context.store is not None:
                await context.store.write(
                    ModelsStoreEntry(
                        models=list(refreshed),
                        checked_at=now_ms(),
                    )
                )

        task = asyncio.create_task(_impl())
        inflight = task
        try:
            await task
        finally:
            inflight = None

    return _refresh
