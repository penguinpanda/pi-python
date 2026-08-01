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

from .._types import Model
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
    
    # DeepSeek Chat
    #
    # 通用对话模型。
    #
    # 特点：
    #
    # • 文本输入
    # • 文本输出
    # • 支持 Tool Calling
    # • 不支持推理(Thinking)
    Model(
        id="deepseek-chat",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek Chat",
        input=["text"],
        output=["text"],
        maxTokens=65536,            # 64K output # 最大输出 Token 数
        thinking=False,             # 模型不会生成推理过程。
        supportsToolCalling=True,   # 支持 Function Calling
        supportsImages=False,       # 不支持图片输入

        # 价格（每百万 Token）。
        #
        # 单位由 Provider 自行约定，
        # 一般与官方 API 定价一致。
        cost={"input": 0.27, "output": 1.10, "cacheRead": 0.07, "cacheWrite": 0.27},
    ),

    # DeepSeek Reasoner
    #
    # 推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 不支持 Tool Calling
    Model(
        id="deepseek-reasoner",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek Reasoner",
        input=["text"],
        output=["text"],
        maxTokens=65536,
        thinking=True, # 模型会生成推理过程。
        supportsToolCalling=False,
        supportsImages=False,
        cost={"input": 0.55, "output": 2.19, "cacheRead": 0.14, "cacheWrite": 0.55},
    ),

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
        maxTokens=384000,           # 最大输出 Token 数
        thinking=True,             # 模型会生成推理过程。
        supportsToolCalling=True,   # 支持 Function Calling
        supportsImages=False,       # 不支持图片输入

        # 价格（每百万 Token）。
        #
        # 单位由 Provider 自行约定，
        # 一般与官方 API 定价一致。
        cost={"input": 0.27, "output": 1.10, "cacheRead": 0.07, "cacheWrite": 0.27},
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
