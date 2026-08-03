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

from ..types import Model, ModelCost, ModelCostTier
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
    # GPT-5 Chat Latest
    #
    # OpenAI 通用对话模型（最新别名）。
    #
    # 特点：
    #
    # • 支持文本输入
    # • 支持图片输入
    # • 支持 Tool Calling
    # • 不输出 Thinking
    # ------------------------------------------------------
    Model(
        id="gpt-5-chat-latest",
        provider="openai",
        api="openai-responses",
        name="GPT-5 Chat Latest",
        input=["text", "image"],
        output=["text"],
        max_tokens=16384,
        context_window=128000,
        cost=ModelCost(input=1.25, output=10.00, cache_read=0.125, cache_write=0.0),
    ),

    # ------------------------------------------------------
    # GPT-5.6 Luna
    #
    # 经济型多模态推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    # • 支持图片输入
    # • 输入超 272K 时价格翻倍
    # ------------------------------------------------------
    Model(
        id="gpt-5.6-luna",
        provider="openai",
        api="openai-responses",
        name="GPT-5.6 Luna",
        input=["text", "image"],
        output=["text"],
        max_tokens=128000,
        context_window=272000,
        reasoning=True,
        cost=ModelCost(
            input=0.20,
            output=1.20,
            cache_read=0.02,
            cache_write=0.25,
            tiers=[
                ModelCostTier(
                    input=0.40,
                    output=1.80,
                    cache_read=0.04,
                    cache_write=0.50,
                    input_tokens_above=272000,
                )
            ],
        ),
    ),

    # ------------------------------------------------------
    # GPT-5.6 Sol
    #
    # 旗舰推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    # • 支持图片输入
    # • 输入超 272K 时价格翻倍
    # ------------------------------------------------------
    Model(
        id="gpt-5.6-sol",
        provider="openai",
        api="openai-responses",
        name="GPT-5.6 Sol",
        input=["text", "image"],
        output=["text"],
        max_tokens=128000,
        context_window=272000,
        reasoning=True,
        cost=ModelCost(
            input=5.00,
            output=30.00,
            cache_read=0.50,
            cache_write=6.25,
            tiers=[
                ModelCostTier(
                    input=10.00,
                    output=45.00,
                    cache_read=1.00,
                    cache_write=12.50,
                    input_tokens_above=272000,
                )
            ],
        ),
    ),

    # ------------------------------------------------------
    # GPT-5.6 Terra
    #
    # 中高端多模态推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    # • 支持图片输入
    # • 输入超 272K 时价格翻倍
    # ------------------------------------------------------
    Model(
        id="gpt-5.6-terra",
        provider="openai",
        api="openai-responses",
        name="GPT-5.6 Terra",
        input=["text", "image"],
        output=["text"],
        max_tokens=128000,
        context_window=272000,
        reasoning=True,
        cost=ModelCost(
            input=2.00,
            output=12.00,
            cache_read=0.20,
            cache_write=2.50,
            tiers=[
                ModelCostTier(
                    input=4.00,
                    output=18.00,
                    cache_read=0.40,
                    cache_write=5.00,
                    input_tokens_above=272000,
                )
            ],
        ),
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
