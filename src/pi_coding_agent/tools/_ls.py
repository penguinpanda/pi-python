"""
ls 工具 — 列出目录内容。
"""

from __future__ import annotations

from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._find import _aborted, _aborted_result
from ._path_utils import resolve_cwd_path

DEFAULT_LIMIT = 500

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list (relative to cwd, defaults to cwd)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of entries to return (default: {DEFAULT_LIMIT})",
        },
    },
    "required": [],
}


def create_ls_tool(cwd: str) -> AgentTool:
    """创建 ls 工具。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        if _aborted(signal):
            return _aborted_result()
        target_str = params.get("path", ".")
        limit = params.get("limit") or DEFAULT_LIMIT
        limit = max(1, int(limit))

        try:
            target = resolve_cwd_path(base, target_str)
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )

        if not target.exists():
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Path not found: {target_str}")],
                details={},
            )

        if target.is_file():
            return AgentToolResult(
                content=[TextContent(type="text", text=_format_entry(target, target.parent))],
                details={"is_file": True},
            )

        # 目录列表
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {target_str}")],
                details={},
            )

        entry_limit_reached = len(entries) > limit
        entries = entries[:limit]
        lines = []
        for entry in entries:
            lines.append(_format_entry(entry, base))

        if not lines:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"(empty directory) {target_str}")],
                details={"count": 0},
            )

        rel = target.relative_to(base) if target.is_relative_to(base) else target
        output = f"Contents of {rel}/:\n" + "\n".join(lines)
        if entry_limit_reached:
            output += f"\n[Truncated: {limit} entries limit]"
        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details={
                "count": len(lines),
                "entryLimitReached": limit if entry_limit_reached else None,
            },
        )

    return AgentTool(
        name="ls",
        prompt_snippet="List directory contents",
        description="List the contents of a directory. Shows files and subdirectories.",
        input_schema=TOOL_SCHEMA,
        label="Ls",
        execute=execute,
    )


def _format_entry(entry: Path, base: Path) -> str:
    """格式化单个目录条目。"""
    try:
        st = entry.stat()
        size = st.st_size
    except OSError:
        size = 0

    # 大小格式化
    if entry.is_dir():
        size_str = "-"
        icon = "📁"
    else:
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}K"
        else:
            size_str = f"{size / (1024 * 1024):.1f}M"
        icon = "📄"

    name = entry.name
    if entry.is_dir():
        name += "/"

    return f"  {icon} {name} ({size_str})"
