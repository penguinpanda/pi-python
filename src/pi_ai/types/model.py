"""
pi_ai.types.model — 模型元数据（Model）与成本。

描述模型的一切元数据：

    Model
    ├── id / provider / api          基础标识
    ├── name                         显示名称
    ├── input / output               模态能力（ModelInput / ModelOutput）
    ├── cost                         Token 成本（ModelCost）
    ├── max_tokens / context_window  容量限制
    ├── base_url / headers           端点覆盖（模型级可覆盖 Provider 级）
    ├── compat                       各 API 兼容配置（ModelCompat）
    └── thinking_level_map           思考级别 → provider 值映射
"""

from dataclasses import dataclass, field

from typing import Literal, TypeAlias

from .compat import ModelCompat
from .common import ThinkingLevelMap


# =========================================================
# 模型基础标识
# =========================================================

ApiId: TypeAlias = str
ProviderId: TypeAlias = str

ModelInput = Literal[
    "text",
    "image"
]

ModelOutput = Literal[
    "text",
]


# =========================================================
# 模型成本（对齐 TS ModelCost）
# =========================================================

@dataclass(slots=True)
class ModelCostRates:
    """$ / 百万 token"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(slots=True)
class ModelCostTier(ModelCostRates):
    """输入用量超过该 token 数时启用此档价格"""

    input_tokens_above: int = 0


@dataclass(slots=True)
class ModelCost(ModelCostRates):
    """请求级价格档位；最高匹配档位适用于整个请求"""

    tiers: list[ModelCostTier] = field(default_factory=list)


@dataclass(slots=True)
class Model:
    """
    模型元数据

    用于描述：

    - 支持哪些能力
    - Token 限制
    - 是否支持图片
    """

    id: str             # 模型唯一 ID
    provider: ProviderId
    api: ApiId            # API 类型
    name: str = ""      # 模型名称
    input: list[ModelInput] = field(default_factory=list)           # 输入能力
    output: list[ModelOutput] = field(default_factory=list)         # 输出能力
    cost: ModelCost = field(default_factory=ModelCost)  # Token 成本
    max_tokens: int = 4096               # 最大 Token
    base_url: str = ""                                   # 模型级 base url（可选覆盖 Provider 级）
    context_window: int = 0                              # 上下文窗口 token 数
    headers: dict[str, str] | None = None               # 模型级自定义头
    compat: ModelCompat | None = None                   # 各 API 的兼容配置
    thinking_level_map: ThinkingLevelMap | None = None  # pi 思考级别 -> provider 值映射
    reasoning: bool = False                             # 是否支持推理（Thinking）


__all__ = [
    "ApiId",
    "ProviderId",
    "ModelInput",
    "ModelOutput",
    "ModelCostRates",
    "ModelCostTier",
    "ModelCost",
    "Model",
]
