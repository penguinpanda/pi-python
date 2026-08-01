"""
bash 工具 — 执行 Shell 命令。

最小核心版: 同步执行 + 输出截断，不支持流式输出/TUI 渲染。
"""

from __future__ import annotations

import asyncio
import os
import sys

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

DEFAULT_TIMEOUT = 120  # 秒
DEFAULT_MAX_BYTES = 50_000

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute",
        },
        "timeout": {
            "type": "integer",
            "description": f"Optional timeout in seconds (default: {DEFAULT_TIMEOUT})",
        },
    },
    "required": ["command"],
}


def create_bash_tool(cwd: str) -> AgentTool:
    """创建 bash 工具。"""
    base = cwd

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = params.get("timeout", DEFAULT_TIMEOUT)

        try:
            result = await _run_command(command, base, timeout)
        except asyncio.TimeoutError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Command timed out after {timeout}s:\n{command}")],
                details={"exit_code": None, "timed_out": True},
            )

        # 截断输出
        output = result["output"]
        truncated = False
        if len(output.encode("utf-8")) > DEFAULT_MAX_BYTES:
            # 保留尾部（更有用）
            output = "(output truncated)\n...\n" + output[-DEFAULT_MAX_BYTES // 2:]
            truncated = True

        exit_code = result["exit_code"]
        canceled = result.get("canceled", False)

        status = "completed"
        if canceled:
            status = "canceled"
        elif exit_code != 0:
            status = f"exit code {exit_code}"

        details = {
            "exit_code": exit_code,
            "truncated": truncated,
            "canceled": canceled,
        }

        if exit_code == 0 and not truncated and not canceled:
            text = output if output.strip() else "(no output)"
        else:
            text = f"[{status}]\n{output}" if output.strip() else f"[{status}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=text)],
            details=details,
        )

    return AgentTool(
        name="bash",
        description="Execute a shell command. Returns stdout, stderr, and exit code.",
        input_schema=TOOL_SCHEMA,
        label="Bash",
        execute=execute,
    )


async def _run_command(
    command: str,
    cwd: str,
    timeout: int,
) -> dict:
    """执行命令，返回 {'output': str, 'exit_code': int | None, 'canceled': bool}。"""
    # 平台感知 shell
    if sys.platform == "win32":
        shell_cmd = ["cmd", "/c", command]
    else:
        shell_cmd = ["bash", "-c", command]

    try:
        proc = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            output = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            return {
                "output": output,
                "exit_code": proc.returncode,
                "canceled": False,
            }
        except asyncio.TimeoutError:
            try:
                proc.terminate()
                try:
                    remaining, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                    remaining, _ = await proc.communicate()
                output = remaining.decode("utf-8", errors="replace") if remaining else ""
                return {
                    "output": output + f"\n[Command timed out after {timeout}s]",
                    "exit_code": proc.returncode,
                    "canceled": True,
                }
            except Exception:
                return {
                    "output": f"[Command timed out after {timeout}s]",
                    "exit_code": None,
                    "canceled": True,
                }
    except FileNotFoundError:
        return {
            "output": f"Error: Shell not found. Command: {command}",
            "exit_code": -1,
            "canceled": False,
        }
    except Exception as e:
        return {
            "output": f"Error executing command: {e}",
            "exit_code": -1,
            "canceled": False,
        }
