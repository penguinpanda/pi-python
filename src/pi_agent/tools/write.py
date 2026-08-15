"""write 工具（对齐 TS harness/tools/write.ts）。"""

from __future__ import annotations

from pi_ai.types import TextContent

from .._types import AgentTool, AgentToolResult
from ..env import get_or_throw
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path


def create_write_tool() -> AgentTool:
    async def execute(
        tool_call_id, params, signal=None, on_update=None, context=None
    ) -> AgentToolResult:
        env = context.env
        path = params["path"]
        content = params["content"]
        absolute_path = await resolve_tool_path(env, path, signal)

        async def _write() -> AgentToolResult:
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")
            get_or_throw(await env.write_file(absolute_path, content, signal))
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")
            byte_count = len(content.encode("utf-8")) if isinstance(content, str) else len(content)
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Successfully wrote {byte_count} bytes to {path}"
                    )
                ],
                details=None,
            )

        return await with_file_mutation_queue(env, absolute_path, _write)

    return AgentTool(
        name="write",
        label="write",
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=["Use write only for new files or complete rewrites."],
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)",
                },
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )
