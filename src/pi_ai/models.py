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

from ._event_stream import AssistantMessageEventStream
from ._types import(
    AssistantMessage,
    Context,
    Model,
    StreamOptions
)
from .auth import InMemoryCredentialStore
from .provider import Provider


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
    def __init__(self) -> None:
        
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
        self._credentials = InMemoryCredentialStore()

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

    # 请求调度

    async def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions  | None = None
    ) -> AssistantMessageEventStream:
        """
        发起流式请求。

        工作流程：

                Model

                │

                ▼

        找到对应 Provider

                │

                ▼

        Provider.stream(...)

                │

                ▼

        AssistantMessageEventStream

        返回后即可：

            async for

        持续读取模型输出。 
        """

        provider = self._require_provider(model.provider)
        return await provider.stream(model, context, options)

    async def complete(
            self,
            model: Model,
            context: Context,
            options: StreamOptions  | None = None
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

    # 凭证管理

    async def set_api_key(self, provider_id: str, api_key: str) -> None:
        """
        保存 Provider 的 API Key。

        保存以后：

        Provider 会优先使用这里保存的 Key，

        而不是环境变量。 
        """

        from .auth import ApiKeyCredential
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