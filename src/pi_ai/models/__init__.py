"""
模型注册表（Models）

=========================================================
模块职责
=========================================================

Models 是整个 SDK 的统一入口（Facade）。

它负责：

    ① Provider 注册

            add_provider()

    ② Model 查询

            get_model()

            get_models()

    ③ 请求调度

            stream()

            complete()

    ④ API Key 管理

            set_api_key()

Models 本身并不会直接调用 OpenAI、DeepSeek 等模型接口。

真正发送 HTTP 请求的是 Provider。

因此：

            Models
               │
               ▼
          找到 Provider
               │
               ▼
      Provider.stream(...)
               │
               ▼
         OpenAI / DeepSeek


=========================================================
整体关系
=========================================================

                    Models
                       │
       ┌───────────────┴───────────────┐
       │                               │
 Provider(OpenAI)              Provider(DeepSeek)
       │                               │
   GPT-4o                       DeepSeek Chat
   GPT-5                        DeepSeek Reasoner

Models 相当于整个 SDK 的调度中心，
负责根据 Model.provider 找到对应 Provider。
"""

import asyncio

from dataclasses import dataclass, field

from ..utils._event_stream import AssistantMessageEventStream
from ..types import AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions
from ..auth import InMemoryCredentialStore
from ..provider import Provider, RefreshModelsContext
from .models_store import (
    InMemoryModelsStore,
    ModelsStore,
    provider_models_store,
)


@dataclass(slots=True)
class ModelsRefreshOptions:
    """Models.refresh 选项。"""

    # False 表示离线/仅缓存初始化（不访问网络）。
    allow_network: bool = True
    # 绕过 provider 新鲜度检查立即抓取。
    force: bool = False
    # 可选中止信号（asyncio.Event）。
    signal: asyncio.Event | None = None


@dataclass(slots=True)
class ModelsRefreshResult:
    """Models.refresh 结果：aborted + 每 provider 的错误（不抛给调用方）。"""

    aborted: bool = False
    errors: dict[str, Exception] = field(default_factory=dict)


class Models:
    """
    模型管理器（Model Registry）。

    整个 SDK 的统一入口。

    主要职责：

    1. 注册 Provider
    2. 查询 Model
    3. 调度请求
    4. 管理 API Key

    Models 本身不实现任何模型接口。

    所有网络请求都会转发给对应 Provider。
    """

    def __init__(
        self,
        *,
        models_store: ModelsStore | None = None,
        credentials=None,
    ) -> None:

        # 已注册的 Provider
        #
        # key:
        #     provider.id
        #
        # value:
        #     Provider
        #
        # 例如：
        #
        # {
        #     "openai": OpenAIProvider(...),
        #     "deepseek": DeepSeekProvider(...)
        # }
        self._providers: dict[str, Provider] = {}

        # 默认使用内存 Credential Store。
        #
        # Provider 获取 API Key 时，
        # 会统一从这里读取。
        #
        # credentials 可注入外部实现（例如 coding-agent 的 AuthStorage），
        # 满足 CredentialStore 接口（read/list/modify/delete）即可。
        self._credentials = credentials or InMemoryCredentialStore()

        # 模型目录持久化（ModelsStore）。
        self._models_store = models_store or InMemoryModelsStore()

    # 模型提供者 管理

    def add_provider(self, provider: Provider) -> None:
        """
        注册一个 Provider。

        如果 Provider ID 已存在，

        则新的 Provider 会覆盖旧 Provider。

        例如：

            add_provider(OpenAIProvider())

        之后即可：

            models.stream(...)
        """

        provider._credential_store = self._credentials
        self._providers[provider.id] = provider

    def get_provider(self, provider_id: str) -> Provider | None:
        """
        根据 Provider ID 获取 Provider。

        例如：

            get_provider("openai")
        """

        return self._providers.get(provider_id)

    def remove_provider(self, provider_id: str) -> None:
        """
        移除一个 Provider。

        如果 Provider 不存在，

        不会抛异常。
        """

        self._providers.pop(provider_id, None)

    def get_providers(self) -> list[Provider]:
        """
        返回所有已注册 Provider。
        """

        return list(self._providers.values())

    # 模型查找

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        """
        获取模型列表。

        provider_id=None

        返回：

            所有 Provider 的模型。

        provider_id="openai"

        返回：

            OpenAI 的模型。
        """

        if provider_id is not None:
            provider = self._providers.get(provider_id)
            return provider.get_models() if provider else []

        result: list[Model] = []

        for provider in self._providers.values():
            result.extend(provider.get_models())
        return result

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        """
        根据：

        Provider ID + Model ID

        查找一个模型。

        例如：

            get_model(
                "openai",
                "gpt-4o"
            )
        """

        provider = self._providers.get(provider_id)
        if not provider:
            return None

        for model in provider.get_models():
            if model.id == model_id:
                return model

        return None

    def get_model_by_id(self, model_id: str) -> Model | None:
        """
        跨所有 Provider 按模型 ID 查找。

        当不关心 Provider 归属时，
        直接按 ID 全局查找。

        例如：

            get_model_by_id("deepseek-chat")

        返回第一个匹配的模型。
        """

        for provider in self._providers.values():
            for model in provider.get_models():
                if model.id == model_id:
                    return model

        return None

    # 动态模型刷新

    async def refresh(
        self,
        options: ModelsRefreshOptions | None = None,
    ) -> ModelsRefreshResult:
        """并发刷新所有支持 refreshModels 的 provider。

        对齐 TS Models.refresh：
        - 单个 provider 失败记录到 errors，不整体抛异常；
        - 失败后以 allow_network=False 重跑一次做缓存恢复（best-effort）；
        - 中止（signal set）不产生错误。
        """
        opts = options or ModelsRefreshOptions()
        result = ModelsRefreshResult()
        refreshable = [
            provider
            for provider in self._providers.values()
            if getattr(provider, "refresh_models", None) is not None
        ]

        async def _refresh_one(provider: Provider) -> None:
            refresh_models = provider.refresh_models
            assert refresh_models is not None
            if opts.signal is not None and opts.signal.is_set():
                result.aborted = True
                return
            store = provider_models_store(self._models_store, provider.id)
            credential = await self._credentials.read(provider.id)
            context = RefreshModelsContext(
                credential=credential,
                store=store,
                allow_network=opts.allow_network,
                force=opts.force,
                signal=opts.signal,
            )
            try:
                await refresh_models(context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if opts.signal is not None and opts.signal.is_set():
                    result.aborted = True
                    return
                result.errors[provider.id] = exc
                # 失败后 best-effort 缓存恢复（离线重跑）。
                try:
                    await refresh_models(
                        RefreshModelsContext(
                            credential=credential,
                            store=store,
                            allow_network=False,
                            force=opts.force,
                            signal=opts.signal,
                        )
                    )
                except Exception:
                    pass

        await asyncio.gather(*(_refresh_one(p) for p in refreshable))
        return result

    # 请求调度

    async def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        """
        发起流式请求。

        工作流程：

                立即返回 EventStream

                后台异步 setup（认证解析 + Provider 调度）

                失败 → error 事件优雅降级（对齐 TS lazyStream）
        """

        from ..api.lazy import lazy_stream

        async def _setup():
            provider = self._require_provider(model.provider)
            return await provider.stream(model, context, options)

        return lazy_stream(model, _setup)

    async def complete(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessage:
        """
        非流式调用。

        内部实际上仍然使用 stream()。

        区别仅在于：

        stream()

        返回：

            EventStream

        complete()

        等待整个 EventStream 完成，

        直接返回最终 AssistantMessage。
        """

        stream = await self.stream(model, context, options)
        return await stream.result()

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """发起 streamSimple 请求（对齐 TS Models.streamSimple）。"""

        from ..api.lazy import lazy_stream

        async def _setup():
            provider = self._require_provider(model.provider)
            return await provider.stream_simple(model, context, options)

        return lazy_stream(model, _setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        """completeSimple：等待整个流结束，返回最终 AssistantMessage。"""
        stream = await self.stream_simple(model, context, options)
        return await stream.result()

    # 凭证管理

    async def set_api_key(self, provider_id: str, api_key: str) -> None:
        """
        保存 Provider 的 API Key。

        保存以后：

        Provider 会优先使用这里保存的 Key，

        而不是环境变量。
        """

        from ..auth import ApiKeyCredential

        await self._credentials.write(provider_id, ApiKeyCredential(key=api_key))

    # 辅助函数

    def _require_provider(self, provider_id: str) -> Provider:
        """
        获取指定 Provider。

        如果 Provider 不存在，

        抛出 ValueError。

        这是一个内部辅助函数，

        避免在多个地方重复判断：

        if provider is None
        """

        provider = self._providers.get(provider_id)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_id}")
        return provider
