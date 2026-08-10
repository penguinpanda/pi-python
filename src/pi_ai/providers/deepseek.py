"""
DeepSeek Provider。

=========================================================
模块职责
=========================================================

本模块负责注册 DeepSeek Provider。

模型元数据不在此手写，统一从自动生成的模型目录加载
（src/pi_ai/models/generated/providers/deepseek.json，
由 scripts/generate_models.py 基于 TS 数据生成），
避免与生成目录形成双数据源。

主要包括：

    ① 加载生成目录中的 DeepSeek 模型

    ② 配置认证方式

    ③ 配置 API 类型

    ④ 配置 Base URL

最终通过：

    deepseek_provider()

创建一个可直接使用的 Provider。
"""

from __future__ import annotations

from ..auth import env_api_key_auth
from ..models.generated import load_generated_models
from ..provider import Provider, create_provider
from ..types import Model


def _load_deepseek_models() -> list[Model]:
    """从自动生成的模型目录加载 DeepSeek 模型（唯一数据源）。"""
    return load_generated_models().get("deepseek", [])


def deepseek_provider(models: list[Model] | None = None) -> Provider:
    """
    创建并返回一个 DeepSeek Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表（默认来自生成目录，可传入自定义列表覆盖）

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
        models=models if models is not None else _load_deepseek_models(),
        # DeepSeek 兼容 OpenAI Chat Completions API。
        api_kind="completions",
        # DeepSeek OpenAI Compatible API 地址。
        base_url="https://api.deepseek.com",
    )
