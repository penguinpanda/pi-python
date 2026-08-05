"""bash 工具（对齐 TS harness/tools/bash.ts）。"""

from __future__ import annotations

from typing import Any

from pi_ai.types import TextContent

from .._types import AgentTool, AgentToolResult
from ..env import get_or_throw
from ..shell_output import execute_shell_with_capture
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size

_MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
_BASH_UPDATE_THROTTLE_MS = 100


class BashToolOptions:
    def __init__(
        self,
        command_prefix: str | None = None,
        prepare=None,
        *,
        session_env_provider=None,
        expose_session_environment: bool = True,
    ) -> None:
        self.command_prefix = command_prefix
        self.prepare = prepare
        self.session_env_provider = session_env_provider
        self.expose_session_environment = expose_session_environment


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {_MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(options: BashToolOptions | None = None) -> AgentTool:
    options = options or BashToolOptions()

    async def execute(
        tool_call_id, params, signal=None, on_update=None, context=None
    ) -> AgentToolResult:
        env = context.env
        command = params["command"]
        timeout = params.get("timeout")
        _validate_timeout(timeout)

        execution_command = (
            f"{options.command_prefix}\n{command}" if options.command_prefix else command
        )
        execution = {
            "command": execution_command,
            "cwd": env.cwd,
            "env": {},
            "inheritEnv": True,
        }
        if options.expose_session_environment and options.session_env_provider is not None:
            session_env = options.session_env_provider()
            if isinstance(session_env, dict):
                execution["env"].update(session_env)
        if options.prepare is not None:
            await options.prepare(execution, context, signal)

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        capture_result = get_or_throw(
            await execute_shell_with_capture(
                env,
                execution["command"],
                {
                    "cwd": execution["cwd"],
                    "env": execution["env"],
                    "inheritEnv": execution["inheritEnv"],
                    "timeout": timeout,
                    "abortSignal": signal,
                    "returnExecutionErrors": True,
                },
            )
        )

        output_text = capture_result.output
        details: Any = None
        if capture_result.truncation.truncated:
            details = {
                "truncation": capture_result.truncation,
                "fullOutputPath": capture_result.full_output_path,
            }
            start_line = (
                capture_result.truncation.total_lines - capture_result.truncation.output_lines + 1
            )
            end_line = capture_result.truncation.total_lines
            if capture_result.truncation.last_line_partial:
                last_line_size = format_size(capture_result.last_line_bytes)
                output_text += (
                    f"\n\n[Showing last {format_size(capture_result.truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {capture_result.full_output_path}]"
                )
            elif capture_result.truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {capture_result.truncation.total_lines}. "
                    f"Full output: {capture_result.full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {capture_result.truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {capture_result.full_output_path}]"
                )

        def _append_status(status: str) -> str:
            return f"{output_text}\n\n{status}" if output_text else status

        if capture_result.cancelled:
            raise ValueError(_append_status("Command aborted"))
        if capture_result.execution_error is not None:
            if capture_result.execution_error.code == "timeout":
                raise ValueError(_append_status(f"Command timed out after {timeout} seconds"))
            raise capture_result.execution_error
        if capture_result.exit_code != 0:
            raise ValueError(_append_status(f"Command exited with code {capture_result.exit_code}"))
        return AgentToolResult(
            content=[TextContent(type="text", text=output_text or "(no output)")],
            details=details,
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds. Prefer commands scoped to the "
            "working directory; do not scan the whole disk (e.g. find /, grep -r /, "
            "locate) and avoid modifying files outside it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["command"],
        },
        execute=execute,
    )
