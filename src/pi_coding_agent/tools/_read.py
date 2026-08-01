"""
read 工具 — 读取文件内容（支持 offset/limit 截断）。
"""

from __future__ import annotations

import os
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

DEFAULT_MAX_LINES = 2000

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the file to read (relative to cwd)",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of lines to read (default: {DEFAULT_MAX_LINES})",
        },
    },
    "required": ["path"],
}


def create_read_tool(cwd: str) -> AgentTool:
    """创建 read 工具。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        file_path = _resolve_path(base, params["path"])
        offset = params.get("offset", 1)
        limit = params.get("limit", DEFAULT_MAX_LINES)

        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: File not found: {params['path']}")],
                details={},
            )
        except UnicodeDecodeError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Cannot read binary file: {params['path']}")],
                details={},
            )
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {params['path']}")],
                details={},
            )

        lines = content.split("\n")
        total_lines = len(lines)

        # offset/limit 钳制
        start = max(0, offset - 1)
        end = min(start + limit, total_lines) if limit > 0 else total_lines
        selected = lines[start:end]

        # 截断提示
        result_text = "\n".join(selected)
        header = ""
        if total_lines > end or offset > 1:
            header = f"[Lines {start + 1}-{end} of {total_lines}]\n"

        return AgentToolResult(
            content=[TextContent(type="text", text=header + result_text)],
            details={"path": str(file_path), "total_lines": total_lines, "offset": offset, "limit": limit},
        )

    return AgentTool(
        name="read",
        description="Read the contents of a file. Supports offset and limit for large files.",
        input_schema=TOOL_SCHEMA,
        label="Read",
        execute=execute,
    )


def _resolve_path(base: Path, file_path: str) -> Path:
    """解析并安全检查文件路径。"""
    p = (base / file_path).resolve()
    # 确保在 cwd 下
    if not str(p).startswith(str(base) + os.sep) and p != base:
        raise ValueError(f"Path traversal detected: {file_path}")
    return p
