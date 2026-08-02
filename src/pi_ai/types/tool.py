"""pi_ai.types.tool — 工具定义（Tool）。

Tool 同时承载：

- 工具定义（name / description / input_schema）—— 模型看到的部分
- 工具实现（handler）—— Python 执行函数
- 生命周期钩子（before_execute / after_execute）—— 可选，供 Agent 层调用
"""

from typing import Any, Awaitable, Callable, Literal

from dataclasses import dataclass

from .common import ConstrainedSamplingConfig


@dataclass(slots=True)
class Tool:
    """
    一个可供模型调用的工具。

    包含：
    - 工具定义（name / description / input_schema）—— 模型看到的部分
    - 工具实现（handler）—— Python 执行函数
    - 生命周期钩子（before_execute / after_execute）—— Agent 层可选调用
    """

    name: str           # 工具名称
    description: str    # 工具说明
    input_schema: dict[str, Any]   # JSON Schema

    # 受约束采样配置（json_schema / grammar）；False 或 None 表示关闭
    constrained_sampling: Literal[False] | ConstrainedSamplingConfig | None = None

    # 工具执行函数（Python 可调用对象）。None 表示仅定义、无实现。
    handler: Callable[..., Awaitable[Any]] | None = None

    # ---- 生命周期钩子（可选，默认 None 不改变现有行为）----

    # 执行前钩子：收到工具调用参数（dict）与执行上下文，
    # 可用于权限检查 / 参数校验 / 记录日志。
    # 返回 None 表示放行；返回 dict 可替换传给 handler 的参数。
    before_execute: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None

    # 执行后钩子：收到 handler 的返回值，可用于后处理 / 记录 / 缓存。
    # 返回 None 保持原结果；返回新值则替换最终结果。
    after_execute: Callable[[Any], Awaitable[Any]] | None = None


__all__ = ["Tool"]
