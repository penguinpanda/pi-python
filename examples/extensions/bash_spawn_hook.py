"""Bash Spawn Hook Example - adjust command / env before execution.

Python port of bash-spawn-hook.ts：用 create_bash_tool 的 spawn_hook，
并以同名工具覆盖内置 bash。
"""

from pathlib import Path

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    from pi_coding_agent.tools import create_bash_tool

    cwd = str(Path.cwd())
    bash_tool = create_bash_tool(
        cwd,
        spawn_hook=lambda ctx: {"PI_SPAWN_HOOK": "1"},
    )
    original_execute = bash_tool.execute

    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        return await original_execute(tool_call_id, params, signal, on_update, ctx)

    pi.register_tool(
        ToolDefinition(
            name="bash",
            label="bash (spawn hook)",
            description=bash_tool.description,
            parameters=bash_tool.input_schema,
            execute=execute,
        )
    )
