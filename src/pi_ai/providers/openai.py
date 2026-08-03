"""
OpenAI Provider。

=========================================================
模块职责
=========================================================

本模块负责注册 OpenAI Provider。

主要包括：

    ① 定义 OpenAI 支持的模型

    ② 配置 API Key 认证方式

    ③ 配置使用的 API 类型（Responses API）

    ④ 配置 OpenAI Base URL

最终通过：

    openai_provider()

创建一个可直接使用的 Provider 实例。

整体流程：

        OPENAI_MODELS
                │
                ▼
      env_api_key_auth()
                │
                ▼
        create_provider()
                │
                ▼
            Provider
"""

from .._types import Model, ModelCost
from ..auth import env_api_key_auth
from ..provider import create_provider, Provider


# ------------------------------------------------------
# OpenAI 支持的模型列表。
#
# 每个 Model 描述模型的静态元数据，
# 包括：
#
#     • 模型 ID
#     • 支持的输入输出
#     • 最大输出 Token
#     • 是否支持 Thinking
#     • 是否支持 Tool Calling
#     • 是否支持图片
#     • Token 价格
#
# Provider 初始化时会直接使用该列表。
# ------------------------------------------------------
OPENAI_MODELS: list[Model] = [

    # ------------------------------------------------------
    # GPT-4o
    #
    # OpenAI 通用多模态模型。
    #
    # 特点：
    #
    # • 支持文本输入
    # • 支持图片输入
    # • 支持 Tool Calling
    # • 不输出 Thinking
    # ------------------------------------------------------
    Model(
        id="gpt-4o",
        provider="openai",
        api="openai-responses",
        name="GPT-4o",
        input=["text", "image"],
        output=["text"],
        max_tokens=16384,
        cost=ModelCost(input=2.50, output=10.00, cache_read=1.25, cache_write=2.50),
    ),

    # ------------------------------------------------------
    # GPT-4o Mini
    #
    # GPT-4o 的轻量版本。
    #
    # 特点：
    #
    # • 更低成本
    # • 更快速度
    # • 支持图片
    # • 支持 Tool Calling
    # ------------------------------------------------------
    Model(
        id="gpt-4o-mini",
        provider="openai",
        api="openai-responses",
        name="GPT-4o Mini",
        input=["text", "image"],
        output=["text"],
        max_tokens=16384,
        cost=ModelCost(input=0.15, output=0.60, cache_read=0.075, cache_write=0.15),
    ),

    # ------------------------------------------------------
    # o4-mini
    #
    # OpenAI 推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    # • 当前仅支持文本输入
    # ------------------------------------------------------
    Model(
        id="o4-mini",
        provider="openai",
        api="openai-responses",
        name="o4 Mini",
        input=["text"],
        output=["text"],
        max_tokens=100000,
        reasoning=True,
        cost=ModelCost(input=1.10, output=4.40, cache_read=0.275, cache_write=1.10),
    ),
]


def openai_provider() -> Provider:
    """
    创建并返回一个 OpenAI Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表

        • API Key 认证

        • Responses API

        • Base URL

    调用者通常只需要：

        provider = openai_provider()

    即可完成 Provider 注册。
    """

    return create_provider(
        id="openai",
        name="OpenAI",
        auth=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"]),
        models=OPENAI_MODELS,

        # 使用 OpenAI Responses API。
        #
        # Responses API 是 OpenAI 新一代统一接口，
        # 支持：
        #
        # • 文本生成
        # • Thinking
        # • Tool Calling
        # • 多模态输入
        api_kind="responses",
        base_url="https://api.openai.com/v1",
    )
