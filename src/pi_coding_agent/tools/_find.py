"""
find 工具 — 搜索文件名（glob 模式）。
"""

from __future__ import annotations

from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._grep import _is_ignored_dir
from ._path_utils import resolve_cwd_path

DEFAULT_MAX_RESULTS = 200

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "The glob pattern to match against file names (e.g., '*.py', '**/test_*.py')",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (relative to cwd, defaults to cwd)",
        },
        "max_results": {
            "type": "integer",
            "description": f"Maximum number of results (default: {DEFAULT_MAX_RESULTS})",
        },
    },
    "required": ["pattern"],
}


def create_find_tool(cwd: str) -> AgentTool:
    """创建 find 工具。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        pattern = params["pattern"]
        search_path_str = params.get("path", ".")
        max_results = params.get("max_results", DEFAULT_MAX_RESULTS)

        try:
            search_path = resolve_cwd_path(base, search_path_str)
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )

        if not search_path.exists():
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Path not found: {search_path_str}")],
                details={},
            )

        results: list[str] = []
        count = 0

        # 支持 ** 递归和普通 glob
        if "**" in pattern:
            iterator = search_path.rglob(pattern.replace("**/", "").replace("**", ""))
            if pattern.startswith("**/"):
                # 递归匹配
                iterator = search_path.rglob(pattern.split("**/", 1)[-1])
        else:
            iterator = search_path.glob(pattern)

        for entry in iterator:
            if count >= max_results:
                break
            if entry.is_file() and not _is_ignored_dir(entry):
                rel_path = entry.relative_to(base) if entry.is_relative_to(base) else entry
                results.append(str(rel_path))
                count += 1

        if not results:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"No files found matching: {pattern}")],
                details={"matches": 0},
            )

        output = f"Found {count} file(s):\n" + "\n".join(sorted(results))
        if count >= max_results:
            output += f"\n[Results truncated at {max_results}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details={"matches": count, "truncated": count >= max_results},
        )

    return AgentTool(
        name="find",
        description="Search for files matching a glob pattern. Returns relative file paths.",
        input_schema=TOOL_SCHEMA,
        label="Find",
        execute=execute,
    )
