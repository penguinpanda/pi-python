"""
write 工具 — 创建或覆盖文件。
"""

from __future__ import annotations

from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._read import _resolve_path

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the file to write (relative to cwd)",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    },
    "required": ["path", "content"],
}


def create_write_tool(cwd: str) -> AgentTool:
    """创建 write 工具。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        try:
            file_path = _resolve_path(base, params["path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = params["content"]
            file_path.write_text(content, encoding="utf-8")
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {params['path']}")],
                details={},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error writing file: {e}")],
                details={},
            )

        return AgentToolResult(
            content=[TextContent(type="text", text=f"File written: {params['path']} ({len(content)} bytes)")],
            details={"path": str(file_path), "bytes": len(content)},
        )

    return AgentTool(
        name="write",
        description="Create or overwrite a file with the given content.",
        input_schema=TOOL_SCHEMA,
        label="Write",
        execute=execute,
    )
