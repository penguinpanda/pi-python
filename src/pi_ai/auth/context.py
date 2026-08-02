"""平台自适应认证上下文（对齐 TS auth/context.ts）。

统一 env / fileExists 两个环境访问点，便于测试注入（FakeEnv）以及
OAuth、Vertex ADC 检测复用。Python 的 os.environ 与 pathlib
天然跨平台，因此默认实现非常简单；接口价值在注入与统一语义。
"""

import os

from pathlib import Path
from typing import Protocol


class AuthContext(Protocol):
    """环境访问接口。"""

    async def env(self, name: str) -> str | None:
        """读取环境变量；空/纯空白视为未设置。"""
        ...

    async def file_exists(self, path: str) -> bool:
        """判断文件是否存在；支持 ~ 前缀展开。"""
        ...


class _DefaultAuthContext:
    """默认实现：os.environ + Path.expanduser().exists()。"""

    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value if isinstance(value, str) and value.strip() else None

    async def file_exists(self, path: str) -> bool:
        try:
            return Path(path).expanduser().exists()
        except OSError:
            return False


def default_auth_context() -> AuthContext:
    """创建默认 AuthContext。"""
    return _DefaultAuthContext()


__all__ = ["AuthContext", "default_auth_context"]
