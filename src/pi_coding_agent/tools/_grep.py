"""
grep 工具 — 搜索文件内容（正则匹配）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._read import _resolve_path

DEFAULT_MAX_RESULTS = 100

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "The regular expression pattern to search for",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in (relative to cwd, defaults to cwd)",
        },
        "include": {
            "type": "string",
            "description": "File glob pattern to include (e.g., '*.py')",
        },
        "max_results": {
            "type": "integer",
            "description": f"Maximum number of results to return (default: {DEFAULT_MAX_RESULTS})",
        },
    },
    "required": ["pattern"],
}


def create_grep_tool(cwd: str) -> AgentTool:
    """创建 grep 工具。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        pattern = params["pattern"]
        search_path = params.get("path", ".")
        include_glob = params.get("include")
        max_results = params.get("max_results", DEFAULT_MAX_RESULTS)

        try:
            target = _resolve_path(base, search_path)
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Invalid regex pattern: {e}")],
                details={},
            )

        results: list[str] = []
        count = 0

        try:
            files = _collect_files(target, include_glob)
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {search_path}")],
                details={},
            )

        for file_path in files:
            if count >= max_results:
                break
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, UnicodeDecodeError, IsADirectoryError):
                continue

            for line_no, line in enumerate(content.splitlines(), 1):
                if count >= max_results:
                    break
                if regex.search(line):
                    rel_path = file_path.relative_to(base) if file_path.is_relative_to(base) else file_path
                    results.append(f"{rel_path}:{line_no}:{line}")
                    count += 1

        if not results:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"No matches found for pattern: {pattern}")],
                details={"matches": 0},
            )

        output = f"Found {count} match(es):\n" + "\n".join(results)
        if count >= max_results:
            output += f"\n[Results truncated at {max_results}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details={"matches": count, "truncated": count >= max_results},
        )

    return AgentTool(
        name="grep",
        description="Search for a regex pattern in files. Returns matching lines with file paths and line numbers.",
        input_schema=TOOL_SCHEMA,
        label="Grep",
        execute=execute,
    )


def _collect_files(target: Path, include_glob: str | None) -> list[Path]:
    """收集要搜索的文件列表。"""
    if target.is_file():
        return [target]

    files: list[Path] = []
    if not target.exists():
        return files

    for entry in target.rglob("*"):
        if entry.is_file():
            if include_glob and not entry.match(include_glob):
                continue
            # 跳过常见忽略目录
            if _is_ignored_dir(entry):
                continue
            files.append(entry)
    return files


def _is_ignored_dir(path: Path) -> bool:
    """检查路径是否在常见忽略目录中。"""
    ignored = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}
    parts = set(path.parts)
    return bool(parts & ignored)
