"""
DeepSeek Provider。

=========================================================
模块职责
=========================================================

本模块负责注册 DeepSeek Provider。

主要包括：

    ① 定义 DeepSeek 支持的模型

    ② 配置认证方式

    ③ 配置 API 类型

    ④ 配置 Base URL

最终通过：

    deepseek_provider()

创建一个可直接使用的 Provider。

整体关系：

        DEEPSEEK_MODELS
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

from ..types import Model, ModelCost
from ..auth import env_api_key_auth
from ..provider import create_provider, Provider


# ------------------------------------------------------
# DeepSeek 支持的模型列表。
#
# 每个 Model 描述模型的元数据，
# 不涉及具体调用逻辑。
#
# Provider 初始化时会直接使用该列表。
# ------------------------------------------------------
DEEPSEEK_MODELS: list[Model] = [
    # # DeepSeek Chat
    # #
    # # 通用对话模型。
    # #
    # # 已从官方定价页下架（Deprecated），保留以兼容旧会话。
    # #
    # # 特点：
    # #
    # # • 文本输入
    # # • 文本输出
    # # • 支持 Tool Calling
    # # • 不支持推理(Thinking)
    # Model(
    #     id="deepseek-chat",
    #     provider="deepseek",
    #     api="openai-completions",
    #     name="DeepSeek Chat",
    #     input=["text"],
    #     output=["text"],
    #     max_tokens=65536,            # 64K output # 最大输出 Token 数
    #     context_window=128000,
    #     deprecated=True,
    #     # 价格（每百万 Token）。
    #     #
    #     # 单位由 Provider 自行约定，
    #     # 一般与官方 API 定价一致。
    #     cost=ModelCost(input=0.27, output=1.10, cache_read=0.07, cache_write=0.27),
    # ),
    # # DeepSeek Reasoner
    # #
    # # 推理模型。
    # #
    # # 已从官方定价页下架（Deprecated），保留以兼容旧会话。
    # #
    # # 特点：
    # #
    # # • 支持 Thinking
    # # • 不支持 Tool Calling
    # Model(
    #     id="deepseek-reasoner",
    #     provider="deepseek",
    #     api="openai-completions",
    #     name="DeepSeek Reasoner",
    #     input=["text"],
    #     output=["text"],
    #     max_tokens=65536,
    #     context_window=65536,
    #     reasoning=True, # 模型会生成推理过程。
    #     deprecated=True,
    #     cost=ModelCost(input=0.55, output=2.19, cache_read=0.14, cache_write=0.55),
    # ),
    # DeepSeek V4 Flash
    #
    # 高速对话模型。
    #
    # 特点：
    # • 高速响应
    # • 支持 Tool Calling
    Model(
        id="deepseek-v4-flash",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek V4 Flash",
        input=["text"],
        output=["text"],
        max_tokens=384000,  # 最大输出 Token 数
        context_window=1000000,
        reasoning=True,  # 支持推理
        # DeepSeek V4 thinking 模式：thinking.type 开关 + reasoning_effort。
        # 官方 effort 映射（deepseek-v4-flash）：
        #   minimal/low -> low，medium/high/xhigh -> high，max -> max；
        # "disabled" 由适配器翻译为 thinking.type=disabled。
        thinking_level_map={
            "off": "disabled",
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "xhigh": "high",
            "max": "max",
        },
        compat={
            "thinkingFormat": "deepseek",
            "requiresReasoningContentOnAssistantMessages": True,
            "supportsLongCacheRetention": False,
            "supportsReasoningEffort": True,
        },
        # 价格（每百万 Token）。
        cost=ModelCost(input=0.14, output=0.28, cache_read=0.0028, cache_write=0.0),
    ),
    # DeepSeek V4 Pro
    #
    # 旗舰推理模型。
    #
    # 特点：
    # • 支持 Thinking
    # • 支持 Tool Calling
    Model(
        id="deepseek-v4-pro",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek V4 Pro",
        input=["text"],
        output=["text"],
        max_tokens=384000,  # 最大输出 Token 数
        context_window=1000000,
        reasoning=True,  # 支持推理
        compat={
            "thinkingFormat": "deepseek",
            "requiresReasoningContentOnAssistantMessages": True,
            "supportsLongCacheRetention": False,
            "supportsReasoningEffort": True,
        },
        cost=ModelCost(input=0.435, output=0.87, cache_read=0.003625, cache_write=0.0),
    ),
]


def deepseek_provider() -> Provider:
    """
    创建并返回一个 DeepSeek Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表

        • API Key 认证

        • API 类型

        • Base URL

    调用者通常只需要：

        provider = deepseek_provider()

    即可完成 Provider 注册。
    """
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        # API Key 优先从：
        #
        # Credential Store
        #
        # 或
        #
        # DEEPSEEK_API_KEY
        #
        # 环境变量读取。
        auth=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"]),
        models=DEEPSEEK_MODELS,
        # DeepSeek 兼容 OpenAI Chat Completions API。
        api_kind="completions",
        # DeepSeek OpenAI Compatible API 地址。
        base_url="https://api.deepseek.com",
    )
