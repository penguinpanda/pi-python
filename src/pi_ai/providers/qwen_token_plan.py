"""
Qwen Token Plan Provider（阿里云百炼 Token Plan）。

=========================================================
模块职责
=========================================================

本模块负责注册 Qwen Token Plan Provider（国际站与中国站）。

阿里云百炼（Model Studio）Token Plan 是订阅制 token 套餐，
使用独立的套餐 API Key（通常为 sk-sp- 前缀）与专用端点，
与按量计费的 DashScope API（见 qwen.py）相互独立。

模型元数据不在此手写，统一从自动生成的模型目录加载
（src/pi_ai/models/generated/providers/qwen-token-plan[-cn].json，
由 scripts/generate_models.py 基于 TS 数据生成），
避免与生成目录形成双数据源。

主要包括：

    ① 加载生成目录中的 Token Plan 模型

    ② 配置 API Key 认证方式

    ③ 配置 API 类型（OpenAI Compatible Completions）

    ④ 配置 Base URL（按区域区分）

最终通过：

    qwen_token_plan_provider()
    qwen_token_plan_cn_provider()

创建两个可直接使用的 Provider 实例（对齐 TS 上游
qwen-token-plan.ts / qwen-token-plan-cn.ts）。

两个区域使用相同的模型目录，但端点与 API Key 不同：

    国际站  https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
    中国站  https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

from ..auth import env_api_key_auth
from ..provider import Provider, create_provider
from ..types import Model

# Token Plan 专用端点（OpenAI 兼容模式）。
QWEN_TOKEN_PLAN_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
QWEN_TOKEN_PLAN_CN_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def qwen_token_plan_provider(models: list[Model] | None = None) -> Provider:
    """
    创建并返回一个 Qwen Token Plan（国际站）Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表（默认空，由 create_default_models() 统一合并生成目录；
          可传入自定义列表覆盖）

        • API Key 认证

        • OpenAI 兼容 Completions API

        • Base URL

    API Key 解析优先级：

        Credential Store（auth.json 中的 "qwen-token-plan"）

            ↓

        QWEN_TOKEN_PLAN_API_KEY 环境变量
    """
    return create_provider(
        id="qwen-token-plan",
        name="Qwen Token Plan",
        auth=env_api_key_auth("Qwen Token Plan API key", ["QWEN_TOKEN_PLAN_API_KEY"]),
        models=models or [],
        api_kind="completions",
        base_url=QWEN_TOKEN_PLAN_BASE_URL,
    )


def qwen_token_plan_cn_provider(models: list[Model] | None = None) -> Provider:
    """
    创建并返回一个 Qwen Token Plan（中国站）Provider。

    与国际站使用相同的模型目录，但端点与 API Key 不同。

    API Key 解析优先级：

        Credential Store（auth.json 中的 "qwen-token-plan-cn"）

            ↓

        QWEN_TOKEN_PLAN_CN_API_KEY 环境变量
    """
    return create_provider(
        id="qwen-token-plan-cn",
        name="Qwen Token Plan CN",
        auth=env_api_key_auth("Qwen Token Plan CN API key", ["QWEN_TOKEN_PLAN_CN_API_KEY"]),
        models=models or [],
        api_kind="completions",
        base_url=QWEN_TOKEN_PLAN_CN_BASE_URL,
    )


__all__ = [
    "qwen_token_plan_provider",
    "qwen_token_plan_cn_provider",
    "QWEN_TOKEN_PLAN_BASE_URL",
    "QWEN_TOKEN_PLAN_CN_BASE_URL",
]
