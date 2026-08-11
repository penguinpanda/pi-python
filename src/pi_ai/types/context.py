"""pi_ai.types.context — 一次模型请求的完整上下文（Context）。

Context 从纯 ChatContext（messages/tools/system_prompt）
增强为携带 Agent Runtime 状态：

    Context
    ├── messages       对话历史
    ├── tools          工具列表
    ├── system_prompt  System Prompt
    ├── metadata       附加元数据（session_id / user_id ...）
    ├── state          运行时状态（Agent 间共享的可变 KV）
    └── trace_id       可观测性 trace 标识（可选）
"""

from dataclasses import dataclass, field

from typing import Any

from .message import Message
from .tool import Tool


@dataclass(slots=True)
class Context:
    """
    一次模型请求的完整上下文。

    最终会发送给 Provider。
    """

    # 对话历史
    messages: list[Message]

    # 工具列表
    tools: list[Tool] = field(default_factory=list)

    # System Prompt
    system_prompt: str | None = None

    # 附加元数据（session_id / user_id 等，供 Agent 与 Provider 使用）
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- Agent Runtime 状态（评审 #4 扩展，可选字段）----

    # 运行时状态（Agent 间共享的可变 KV；不发送给 Provider）
    state: dict[str, Any] = field(default_factory=dict)

    # 可观测性 trace 标识（可选；关联 Trace 记录）
    trace_id: str | None = None


__all__ = ["Context"]
