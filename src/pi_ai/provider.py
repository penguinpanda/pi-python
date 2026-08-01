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

from dataclasses import dataclass, field
from typing import Literal

from ._event_stream import AssistantMessageEventStream
from ._types import (
    AssistantMessage,
    Context,
    Model,
    StreamOptions,
)

from .api.completions import chat_completions_stream
from .api.responses import responses_stream
from .auth import EnvApiKeyAuth, InMemoryCredentialStore, resolve_api_key

# Provider 使用的 API 类型。
#
# completions
#     Chat Completions API
#
# responses
#     Responses API

ApiKind = Literal["completions", "responses"]

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
    auth: EnvApiKeyAuth

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

    def get_models(self) -> list[Model]:
        """
        返回 Provider 支持的所有模型。

        返回副本，

        避免调用者修改内部列表。
        """

        return list(self.models)

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

        # 解析 API Key。
        #
        # 优先级：
        #
        # StreamOptions.apiKey（本次请求覆盖）
        #
        # ↓
        #
        # Credential Store
        #
        # ↓
        #
        # Environment Variable
        opts = options or {}
        api_key = opts.get("apiKey") or await resolve_api_key(self.auth, self._credential_store, self.id)
        base_url = self.base_url or ""

        # 根据 Provider 使用的 API 类型，
        # 调度到不同实现。
        #
        # 不同 Provider 可以共享同一套 Models，
        # 但底层 API 实现不同。
        if self._api_kind == "responses":
            return await responses_stream(model, context, api_key, base_url, options)
        else:
            return await chat_completions_stream(model, context, api_key, base_url, options)

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
        auth: EnvApiKeyAuth,
        models: list[Model],
        api_kind: ApiKind = "completions",
        base_url: str | None = None,
) -> Provider:
    """
    创建 Provider。

    相比直接调用 Provider(...)

    提供：

    - 默认值
    - 更简单的参数

    主要用于：

    Provider 注册。
    """

    return Provider(
        id=id,
        name=name or id,
        auth=auth,
        models=models,
        _api_kind=api_kind,
        base_url=base_url
    )