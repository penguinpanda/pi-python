"""内置工具（Phase 4.3）：read / write / edit / bash。

所有工具通过 ExecutionEnv 进行 I/O，平台无关；execute 接收
(tool_call_id, params, signal, on_update, context) 五参（对齐 AgentHarnessTool）。
"""

from .bash import BashToolOptions, create_bash_tool
from .edit import create_edit_tool
from .read import ReadToolOptions, create_read_tool
from .tool_context import ExecutionToolContext
from .write import create_write_tool

__all__ = [
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "ExecutionToolContext",
    "ReadToolOptions",
    "BashToolOptions",
]
