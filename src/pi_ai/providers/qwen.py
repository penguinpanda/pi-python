"""
Qwen Provider（阿里云 DashScope，OpenAI 兼容模式）。

=========================================================
模块职责
=========================================================

本模块负责注册 Qwen Provider。

主要包括：

    ① 定义 Qwen 支持的模型

    ② 配置 API Key 认证方式

    ③ 配置 API 类型（OpenAI Compatible Completions）

    ④ 配置 Base URL

最终通过：

    qwen_provider()

创建一个可直接使用的 Provider 实例。

DashScope 提供 OpenAI 兼容接口：

    https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

API Key 在阿里云百炼控制台获取。
"""

from __future__ import annotations

from ..types import Model, ModelCost
from ..auth import env_api_key_auth
from ..provider import create_provider, Provider


# ------------------------------------------------------
# Qwen 支持的模型列表。
#
# 价格按 DashScope 公开定价换算为美元（$/百万 token），
# 仅供参考，可按实际账单调整。
# ------------------------------------------------------
QWEN_MODELS: list[Model] = [
    Model(
        id="qwen-turbo",
        provider="qwen",
        api="openai-completions",
        name="Qwen Turbo",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=131072,
        cost=ModelCost(input=0.042, output=0.084, cache_read=0.0042, cache_write=0.0),
    ),
    Model(
        id="qwen-plus",
        provider="qwen",
        api="openai-completions",
        name="Qwen Plus",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=131072,
        cost=ModelCost(input=0.112, output=0.28, cache_read=0.0112, cache_write=0.0),
    ),
    Model(
        id="qwen-max",
        provider="qwen",
        api="openai-completions",
        name="Qwen Max",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=32768,
        cost=ModelCost(input=0.336, output=1.344, cache_read=0.0336, cache_write=0.0),
    ),
    Model(
        id="qwen3-235b-a22b",
        provider="qwen",
        api="openai-completions",
        name="Qwen3 235B A22B",
        input=["text"],
        output=["text"],
        max_tokens=131072,
        context_window=262144,
        reasoning=True,
        cost=ModelCost(input=0.112, output=0.448, cache_read=0.0112, cache_write=0.0),
    ),
    Model(
        id="qwen3-30b-a3b",
        provider="qwen",
        api="openai-completions",
        name="Qwen3 30B A3B",
        input=["text"],
        output=["text"],
        max_tokens=32768,
        context_window=131072,
        reasoning=True,
        cost=ModelCost(input=0.042, output=0.14, cache_read=0.0042, cache_write=0.0),
    ),
    Model(
        id="qwen3-vl-flash",
        provider="qwen",
        api="openai-completions",
        name="Qwen3 VL Flash",
        input=["text", "image"],
        output=["text"],
        max_tokens=8192,
        context_window=32768,
        cost=ModelCost(input=0.042, output=0.14, cache_read=0.0042, cache_write=0.0),
    ),
    Model(
        id="qwen-vl-plus",
        provider="qwen",
        api="openai-completions",
        name="Qwen VL Plus",
        input=["text", "image"],
        output=["text"],
        max_tokens=4096,
        context_window=32768,
        cost=ModelCost(input=0.21, output=0.63, cache_read=0.021, cache_write=0.0),
    ),
    Model(
        id="qwen-vl-max",
        provider="qwen",
        api="openai-completions",
        name="Qwen VL Max",
        input=["text", "image"],
        output=["text"],
        max_tokens=8192,
        context_window=32768,
        cost=ModelCost(input=2.8, output=8.4, cache_read=0.28, cache_write=0.0),
    ),
]


def qwen_provider() -> Provider:
    """
    创建并返回一个 Qwen Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表

        • API Key 认证

        • OpenAI 兼容 Completions API

        • Base URL

    API Key 解析优先级：

        Credential Store（pi login 保存的凭证）

            ↓

        DASHSCOPE_API_KEY / QWEN_API_KEY 环境变量
    """
    return create_provider(
        id="qwen",
        name="Qwen",

        # API Key 优先从 Credential Store，其次环境变量读取。
        auth=env_api_key_auth(
            "Qwen (DashScope) API key",
            ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        ),
        models=QWEN_MODELS,

        # DashScope OpenAI 兼容模式使用 Chat Completions API。
        api_kind="completions",

        # DashScope OpenAI Compatible API 地址（/v1 结尾，
        # 客户端会自动追加 /chat/completions）。
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


__all__ = ["qwen_provider", "QWEN_MODELS"]
