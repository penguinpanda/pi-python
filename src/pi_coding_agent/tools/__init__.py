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
import os

from pi_agent import AgentTool
from pi_agent.tools import BashToolOptions
from pi_agent.env import FileError, PythonExecutionEnv
from pi_agent.shell_output import execute_shell_with_capture
from pi_agent.tools.file_mutation_queue import with_file_mutation_queue
from pi_agent.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from pi_agent.tools import (
    ReadToolOptions,
    create_bash_tool as _create_pi_bash_tool,
    create_edit_tool as _create_pi_edit_tool,
    create_read_tool as _create_pi_read_tool,
    create_write_tool as _create_pi_write_tool,
)
from pi_agent.tools.image_pipeline import process_image

from .._config import get_bin_dir
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
        prompt_snippet=tool.prompt_snippet,
        prompt_guidelines=tool.prompt_guidelines,
        execute=execute,
    )


def create_read_tool(cwd: str) -> AgentTool:
    """创建 read 工具（复用 pi_agent 实现，绑定本地执行环境）。"""
    tool = _bind_env(
        _create_pi_read_tool(ReadToolOptions(image_processor=process_image)),
        PythonExecutionEnv(cwd),
    )
    tool.description += (
        " Only operate on files inside the current working directory. If a file is "
        "not found, report that to the user; do not search the whole disk "
        "(e.g. find /, grep -r /, locate)."
    )
    original_execute = tool.execute

    async def execute(tool_call_id, params, signal=None, on_update=None, context=None):
        try:
            return await original_execute(tool_call_id, params, signal, on_update, context)
        except FileError as exc:
            if exc.code == "not_found":
                raise ValueError(
                    f"{exc} The file is not in the working directory. Report this to "
                    "the user directly; do not search the whole disk "
                    "(e.g. find /, grep -r /, locate)."
                ) from exc
            raise

    tool.execute = execute
    return tool


def create_write_tool(cwd: str) -> AgentTool:
    """创建 write 工具（复用 pi_agent 实现，绑定本地执行环境）。

    写入限制在工作目录内（restrict_paths_to_cwd）；read 保持全盘可读。
    """
    return _bind_env(_create_pi_write_tool(), PythonExecutionEnv(cwd, restrict_paths_to_cwd=True))


def create_edit_tool(cwd: str) -> AgentTool:
    """创建 edit 工具（复用 pi_agent 实现，绑定本地执行环境）。

    写入限制在工作目录内（restrict_paths_to_cwd）。
    """
    return _bind_env(_create_pi_edit_tool(), PythonExecutionEnv(cwd, restrict_paths_to_cwd=True))


def create_bash_tool(
    cwd: str,
    *,
    session_env_provider=None,
    expose_session_environment: bool = True,
    spawn_hook=None,
) -> AgentTool:
    """创建 bash 工具（复用 pi_agent 实现，绑定本地执行环境）。

    session_env_provider 返回注入子进程的会话环境变量（PI_SESSION_ID 等）；
    spawn_hook({command, cwd, env}) 返回 {command, cwd, env} 可重写任一字段
    （对齐 TS createBashTool 的 spawnHook；仅返回额外环境变量 dict 也可）。
    """

    def _prepend_bin_dir_to_path(env: dict[str, str]) -> None:
        """把 pi bin 目录前置到 PATH（对齐 TS getShellEnv，PATH 键大小写不敏感）。"""
        bin_dir = str(get_bin_dir())
        env_path_key = next((key for key in env if key.lower() == "path"), None)
        if env_path_key is not None:
            current = env[env_path_key]
            env[env_path_key] = bin_dir + (os.pathsep + current if current else "")
            return
        base_key = next((key for key in os.environ if key.lower() == "path"), "PATH")
        current = os.environ.get(base_key, "")
        env[base_key] = bin_dir + (os.pathsep + current if current else "")

    async def _prepare(execution, context, signal) -> None:
        _prepend_bin_dir_to_path(execution["env"])
        if spawn_hook is not None:
            # 对齐 TS BashSpawnHook：返回 {command, cwd, env} 可重写任一字段；
            # 不含这三个键的 dict 视为 env 直接合并（兼容旧调用方）。
            hook_context = {
                "command": execution["command"],
                "cwd": execution["cwd"],
                "env": dict(execution["env"]),
            }
            result = spawn_hook(hook_context)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                if any(key in result for key in ("command", "cwd", "env")):
                    if isinstance(result.get("command"), str):
                        execution["command"] = result["command"]
                    if isinstance(result.get("cwd"), str):
                        execution["cwd"] = result["cwd"]
                    env = result.get("env")
                    if isinstance(env, dict):
                        execution["env"] = {
                            key: value for key, value in env.items() if value is not None
                        }
                else:
                    execution["env"].update(
                        {key: value for key, value in result.items() if value is not None}
                    )

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


# ---------------------------------------------------------------------------
# ToolDefinition compatibility helpers
# ---------------------------------------------------------------------------


def _tool_to_definition(tool: AgentTool) -> dict:
    return {
        "name": tool.name,
        "label": tool.label,
        "description": tool.description,
        "parameters": tool.input_schema,
        "prompt_snippet": tool.prompt_snippet,
        "prompt_guidelines": tool.prompt_guidelines,
        "execution_mode": getattr(tool, "execution_mode", "parallel"),
        "execute": tool.execute,
    }


def create_read_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_read_tool(cwd))


def create_write_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_write_tool(cwd))


def create_edit_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_edit_tool(cwd))


def create_bash_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_bash_tool(cwd, **(options or {})))


def create_grep_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_grep_tool(cwd))


def create_find_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_find_tool(cwd))


def create_ls_tool_definition(cwd: str, options: dict | None = None):
    return _tool_to_definition(create_ls_tool(cwd))


def create_local_bash_operations(cwd: str):
    """返回 TS BashOperations 兼容的本地 exec 实现。"""

    env = PythonExecutionEnv(cwd)

    class LocalBashOperations:
        async def exec(self, command, cwd_override=None, options=None):
            options = dict(options or {})
            ok, result = await execute_shell_with_capture(
                env,
                command,
                {
                    "cwd": cwd_override or cwd,
                    "inheritEnv": True,
                    "timeout": options.get("timeout"),
                    "abortSignal": options.get("abortSignal"),
                    "onChunk": options.get("onChunk"),
                    "returnExecutionErrors": True,
                },
            )
            if not ok:
                raise result
            if result.execution_error is not None:
                raise result.execution_error
            return {
                "output": result.output,
                "exitCode": result.exit_code,
                "cancelled": result.cancelled,
                "truncated": result.truncated,
                "fullOutputPath": result.full_output_path,
            }

    return LocalBashOperations()


class TruncationOptions(dict):
    """Truncation options placeholder (TS TruncationOptions)."""


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
    # Tool definitions / operations / truncation (TS core/tools exports)
    "create_read_tool_definition",
    "create_write_tool_definition",
    "create_edit_tool_definition",
    "create_bash_tool_definition",
    "create_grep_tool_definition",
    "create_find_tool_definition",
    "create_ls_tool_definition",
    "create_local_bash_operations",
    "with_file_mutation_queue",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "TruncationOptions",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]
