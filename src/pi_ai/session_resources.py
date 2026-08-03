"""pi_ai.session_resources — Session 资源清理（对齐 TS packages/ai/src/session-resources.ts）。

全局注册表：持有会话级资源的模块（WebSocket / SSE 连接、文件句柄等）把清理
函数注册进来；session close / reload / shutdown 统一调用
cleanup_session_resources()。单个清理失败不影响其它清理，
错误收集后统一抛出 CleanupError。
"""

from typing import Callable

# 清理函数：session_id | None
SessionResourceCleanup = Callable[[str | None], None]

_cleanups: set[SessionResourceCleanup] = set()


class CleanupError(Exception):
    """资源清理聚合错误（Python <3.11 无 ExceptionGroup 时的替代）。"""

    def __init__(self, errors: list[BaseException]) -> None:
        super().__init__("Failed to cleanup session resources")
        self.errors = errors


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """注册清理函数；返回反注册函数。"""
    _cleanups.add(cleanup)

    def _unregister() -> None:
        _cleanups.discard(cleanup)

    return _unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    """执行全部已注册清理；收集错误后统一抛出。"""
    errors: list[BaseException] = []
    for cleanup in list(_cleanups):
        try:
            cleanup(session_id)
        except BaseException as exc:  # noqa: BLE001 — 聚合而非中断
            errors.append(exc)
    if errors:
        raise CleanupError(errors)


__all__ = [
    "SessionResourceCleanup",
    "CleanupError",
    "register_session_resource_cleanup",
    "cleanup_session_resources",
]
