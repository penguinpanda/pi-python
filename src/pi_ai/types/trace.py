"""pi_ai.types.trace — 可观测性（Trace / TraceSpan）。

类型先行：本模块只定义数据结构，不提供运行时实现。

用途：

- LangSmith / OpenTelemetry 对接
- Agent 调试（trace_id 贯穿 Context → 请求 → 事件）
- 性能分析（span 开始/结束时间）
"""

from dataclasses import dataclass, field

from typing import Any, NotRequired, TypedDict


class TraceSpan(TypedDict, total=False):
    """单个 span 记录（一段可观测的操作）。"""

    name: str                        # span 名称（如 "llm.call" / "tool.execute"）
    parent_id: NotRequired[str]      # 父 span 标识（构成树状结构）
    start_time: int                  # 开始时间（Unix 毫秒）
    end_time: NotRequired[int]       # 结束时间（Unix 毫秒）
    status: NotRequired[str]         # 状态（ok / error / cancelled）
    metadata: NotRequired[dict[str, Any]]  # 附加信息


@dataclass(slots=True)
class Trace:
    """一次完整追踪（trace）记录。"""

    trace_id: str                    # 追踪唯一标识
    parent_id: str | None = None     # 父追踪标识（多 Agent 嵌套时）
    name: str = ""                   # 追踪名称（如 "agent.run"）
    start_time: int = 0              # 开始时间（Unix 毫秒）
    end_time: int | None = None      # 结束时间（Unix 毫秒）
    spans: list[TraceSpan] = field(default_factory=list)

    def add_span(self, span: TraceSpan) -> None:
        """记录一个 span。"""
        self.spans.append(span)

    def finish(self, end_time: int) -> None:
        """标记追踪结束。"""
        self.end_time = end_time


__all__ = ["Trace", "TraceSpan"]
