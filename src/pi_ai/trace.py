"""pi_ai.trace — 可观测性运行时（对齐 types/trace.py 的类型定义）。

类型定义在 `pi_ai.types.trace`（仅类型，无运行时）；本模块提供运行时实现：

- TraceSpanHandle：进行中 span 的句柄（context manager，退出时写入 Trace）；
- TraceTracer：运行时追踪器（start_span / finish / 嵌套 span）；
- current_trace / set_current_trace / reset_current_trace / run_with_trace：
  contextvar 传播，trace_id 贯穿 Context → 请求 → 事件；
- trace_to_dict / trace_from_dict / trace_duration_ms：序列化与时长辅助。

用途：

- LangSmith / OpenTelemetry 对接
- Agent 调试（trace_id 贯穿 Context → 请求 → 事件）
- 性能分析（span 开始/结束时间）
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .types.common import now_ms
from .types.trace import Trace, TraceSpan


def _new_id() -> str:
    """span/trace ID：时间排序 UUID v7（延迟导入避免与 utils 循环）。"""
    from .utils.uuid import uuidv7

    return uuidv7()


@dataclass(slots=True)
class TraceSpanHandle:
    """进行中 span 的句柄；作为 context manager 使用时退出即自动落盘。"""

    tracer: "TraceTracer"
    name: str
    start_time: int
    span_id: str
    parent_id: str | None = None
    metadata: dict[str, Any] | None = None
    status: str = "ok"
    _ended: bool = field(default=False, init=False, repr=False)
    _last_span: TraceSpan | None = field(default=None, init=False, repr=False)

    def end(
        self,
        status: str = "ok",
        end_time: int | None = None,
    ) -> TraceSpan:
        """结束 span 并写入所属 Trace；重复调用返回第一次的结果。"""
        if self._ended:
            assert self._last_span is not None
            return self._last_span
        self._ended = True
        span: TraceSpan = {
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": end_time if end_time is not None else now_ms(),
            "status": status,
            "metadata": self.metadata,
        }
        self.status = status
        assert self.tracer.trace is not None
        self.tracer.trace.add_span(span)
        self._last_span = span
        return span

    def __enter__(self) -> "TraceSpanHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end(status="error" if exc_type is not None else self.status)


@dataclass(slots=True)
class TraceTracer:
    """运行时追踪器：创建 span 并产出完整 Trace。"""

    name: str = ""
    trace: Trace | None = None
    parent_id: str | None = None
    _next_span_id: int = field(default=0, init=False, repr=False)
    _token: Token | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = Trace(
                trace_id=_new_id(),
                name=self.name,
                start_time=now_ms(),
            )

    def start_span(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        start_time: int | None = None,
    ) -> TraceSpanHandle:
        """开始一个 span；结束（end / with 退出）后写入 self.trace。"""
        assert self.trace is not None
        span_id = f"{self.trace.trace_id}-{self._next_span_id}"
        self._next_span_id += 1
        return TraceSpanHandle(
            tracer=self,
            name=name,
            start_time=start_time if start_time is not None else now_ms(),
            span_id=span_id,
            parent_id=self.parent_id,
            metadata=metadata,
        )

    def finish(self, end_time: int | None = None) -> Trace:
        """结束追踪；返回最终 Trace。"""
        assert self.trace is not None
        self.trace.finish(end_time if end_time is not None else now_ms())
        return self.trace

    def __enter__(self) -> "TraceTracer":
        self._token = _current_trace.set(self.trace)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish()
        if self._token is not None:
            _current_trace.reset(self._token)
            self._token = None


# ---------------------------------------------------------------------------
# 当前 trace 传播（contextvar）
# ---------------------------------------------------------------------------

_current_trace: ContextVar[Trace | None] = ContextVar("pi_ai_current_trace", default=None)


def current_trace() -> Trace | None:
    """当前上下文中的 Trace；未在 trace 上下文内返回 None。"""
    return _current_trace.get()


def set_current_trace(trace: Trace | None) -> Token:
    """显式设置当前 Trace；返回 token 用于 reset。"""
    return _current_trace.set(trace)


def reset_current_trace(token: Token) -> None:
    """恢复 set_current_trace 之前的 Trace 上下文。"""
    _current_trace.reset(token)


_T = TypeVar("_T")


def run_with_trace(
    trace: Trace | None,
    fn: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    """在指定 Trace 上下文中执行 fn（异步任务会自动捕获该 context）。"""
    token = _current_trace.set(trace)
    try:
        return fn(*args, **kwargs)
    finally:
        _current_trace.reset(token)


# ---------------------------------------------------------------------------
# 序列化与时长辅助
# ---------------------------------------------------------------------------


def trace_duration_ms(trace: Trace) -> int | None:
    """追踪持续时长（毫秒）；未结束或未开始返回 None。"""
    if trace.end_time is None or trace.start_time == 0:
        return None
    return max(0, trace.end_time - trace.start_time)


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    """Trace → JSON 可序列化字典。"""
    return {
        "trace_id": trace.trace_id,
        "parent_id": trace.parent_id,
        "name": trace.name,
        "start_time": trace.start_time,
        "end_time": trace.end_time,
        "spans": list(trace.spans),
    }


def trace_from_dict(data: dict[str, Any]) -> Trace:
    """字典 → Trace（trace_to_dict 的逆操作）。"""
    return Trace(
        trace_id=data["trace_id"],
        parent_id=data.get("parent_id"),
        name=data.get("name", ""),
        start_time=data.get("start_time", 0),
        end_time=data.get("end_time"),
        spans=list(data.get("spans") or []),
    )


__all__ = [
    "Trace",
    "TraceSpan",
    "TraceSpanHandle",
    "TraceTracer",
    "current_trace",
    "set_current_trace",
    "reset_current_trace",
    "run_with_trace",
    "trace_duration_ms",
    "trace_to_dict",
    "trace_from_dict",
]
