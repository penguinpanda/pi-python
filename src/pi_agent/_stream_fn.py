"""
默认流函数注册（依赖注入）

通过模块级单例让核心包不依赖具体 provider SDK。
宿主应用在启动时调用 set_default_stream_fn(models.stream)，
Agent 构造时若未显式传 stream_fn 则回退到默认值。
"""

from __future__ import annotations

from ._types import StreamFn

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """注册全局默认流函数。传 None 可清除。"""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    """获取全局默认流函数。未设置时抛 RuntimeError。"""
    if _default_stream_fn is None:
        raise RuntimeError("Pass streamFn explicitly or call setDefaultStreamFn().")
    return _default_stream_fn
