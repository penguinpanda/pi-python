"""内置编码工具集合。

read / write / edit / bash 复用 pi_agent 的工具实现（基于 ExecutionEnv，
支持 Result 错误编码、中断信号、流式输出与完整截断策略），此处仅把它们
绑定到 coding-agent 的工作目录执行环境；grep / find / ls 为 coding-agent
特有工具（pi_agent 无对应实现）。

提供 7 个编码工具: read, write, edit, bash, grep, find, ls
以及 3 种组合器与工具过滤: create_all_tools, create_coding_tools,
create_readonly_tools, filter_tools_by_names
"""

from __future__ import annotations

import inspect

from pi_agent import AgentTool
from pi_agent.tools import BashToolOptions
from pi_agent.env import PythonExecutionEnv
from pi_agent.tools import (
    create_bash_tool as _create_pi_bash_tool,
    create_edit_tool as _create_pi_edit_tool,
    create_read_tool as _create_pi_read_tool,
    create_write_tool as _create_pi_write_tool,
)

from ._find import create_find_tool
from ._grep import create_grep_tool
from ._ls import create_ls_tool


class _LocalToolContext:
    """工具执行上下文：把 coding-agent 的本地执行环境作为第 5 参 context 注入。"""

    __slots__ = ("env",)

    def __init__(self, env: PythonExecutionEnv) -> None:
        self.env = env


def _bind_env(tool: AgentTool, env: PythonExecutionEnv) -> AgentTool:
    """把 pi_agent 工具绑定到本地 ExecutionEnv。

    最小 agent loop 调用 execute 时只传 4 参（无 context），pi_agent 工具
    需要第 5 参 context.env，因此这里补齐。
    """
    original_execute = tool.execute

    async def execute(tool_call_id, params, signal=None, on_update=None, context=None):
        if context is None or getattr(context, "env", None) is None:
            context = _LocalToolContext(env)
        return await original_execute(tool_call_id, params, signal, on_update, context)

    return AgentTool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        label=tool.label,
        execute=execute,
    )


def create_read_tool(cwd: str) -> AgentTool:
    """创建 read 工具（复用 pi_agent 实现，绑定本地执行环境）。"""
    return _bind_env(_create_pi_read_tool(), PythonExecutionEnv(cwd))


def create_write_tool(cwd: str) -> AgentTool:
    """创建 write 工具（复用 pi_agent 实现，绑定本地执行环境）。"""
    return _bind_env(_create_pi_write_tool(), PythonExecutionEnv(cwd))


def create_edit_tool(cwd: str) -> AgentTool:
    """创建 edit 工具（复用 pi_agent 实现，绑定本地执行环境）。"""
    return _bind_env(_create_pi_edit_tool(), PythonExecutionEnv(cwd))


def create_bash_tool(
    cwd: str,
    *,
    session_env_provider=None,
    expose_session_environment: bool = True,
    spawn_hook=None,
) -> AgentTool:
    """创建 bash 工具（复用 pi_agent 实现，绑定本地执行环境）。

    session_env_provider 返回注入子进程的会话环境变量（PI_SESSION_ID 等）；
    spawn_hook(ctx) 返回额外环境变量（对齐 TS createBashTool 的 spawnHook）。
    """

    async def _prepare(execution, context, signal) -> None:
        if spawn_hook is not None:
            extra = spawn_hook(context)
            if inspect.isawaitable(extra):
                extra = await extra
            if isinstance(extra, dict):
                execution["env"].update(extra)

    return _bind_env(
        _create_pi_bash_tool(
            BashToolOptions(
                prepare=_prepare,
                session_env_provider=session_env_provider,
                expose_session_environment=expose_session_environment,
            )
        ),
        PythonExecutionEnv(cwd),
    )


def create_all_tools(
    cwd: str,
    *,
    bash_session_env_provider=None,
    bash_expose_session_environment: bool = True,
    bash_spawn_hook=None,
) -> list:
    """全部 7 个工具。"""
    return [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(
            cwd,
            session_env_provider=bash_session_env_provider,
            expose_session_environment=bash_expose_session_environment,
            spawn_hook=bash_spawn_hook,
        ),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_coding_tools(
    cwd: str,
    *,
    bash_session_env_provider=None,
    bash_expose_session_environment: bool = True,
    bash_spawn_hook=None,
) -> list:
    """编码模式: read + bash + edit + write。"""
    return [
        create_read_tool(cwd),
        create_bash_tool(
            cwd,
            session_env_provider=bash_session_env_provider,
            expose_session_environment=bash_expose_session_environment,
            spawn_hook=bash_spawn_hook,
        ),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_readonly_tools(cwd: str) -> list:
    """只读探索: read + grep + find + ls。"""
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def filter_tools_by_names(
    tools: list[AgentTool],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[AgentTool]:
    """工具白名单 / 黑名单过滤。"""
    if include is not None:
        include_set = set(include)
        tools = [tool for tool in tools if tool.name in include_set]
    if exclude:
        exclude_set = set(exclude)
        tools = [tool for tool in tools if tool.name not in exclude_set]
    return tools


__all__ = [
    "create_all_tools",
    "create_coding_tools",
    "create_readonly_tools",
    "filter_tools_by_names",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
]
