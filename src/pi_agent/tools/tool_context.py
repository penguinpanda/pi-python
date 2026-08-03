"""工具执行上下文（对齐 TS harness/tools/tool-context.ts）。"""

from __future__ import annotations

from typing import Protocol

from ..env import ExecutionEnv


class ExecutionToolContext(Protocol):
    """内置执行工具所需的文件系统与 shell 上下文。"""

    env: ExecutionEnv
