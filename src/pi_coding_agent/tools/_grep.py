"""
grep 工具 — 搜索文件内容（正则匹配，对齐 TS core/tools/grep.ts schema）。

schema 与 TS 一致：pattern / path / glob / ignoreCase / literal / context / limit。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._path_utils import resolve_cwd_path

DEFAULT_LIMIT = 100
GREP_MAX_LINE_LENGTH = 1000

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Search pattern (regex or literal string)",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
        },
        "ignoreCase": {
            "type": "boolean",
            "description": "Case-insensitive search (default: false)",
        },
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "integer",
            "description": "Number of lines to show before and after each match (default: 0)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of matches to return (default: {DEFAULT_LIMIT})",
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
        signal: asyncio.Event | None = None,
        on_update: object = None,
    ) -> AgentToolResult:
        pattern = params["pattern"]
        search_path = params.get("path", ".")
        # glob 优先（对齐 TS）；兼容旧的 include 字段名。
        glob_pattern = params.get("glob") or params.get("include")
        ignore_case = bool(params.get("ignoreCase"))
        literal = bool(params.get("literal"))
        context = params.get("context") or 0
        limit = params.get("limit") or params.get("max_results") or DEFAULT_LIMIT
        limit = max(1, int(limit))
        context = max(0, int(context))

        if signal is not None and signal.is_set():
            return AgentToolResult(
                content=[TextContent(type="text", text="Operation aborted")],
                details={"aborted": True},
            )

        try:
            target = resolve_cwd_path(base, search_path)
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )

        try:
            if literal:
                regex = re.compile(re.escape(pattern), re.IGNORECASE if ignore_case else 0)
            else:
                regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Invalid regex pattern: {e}")],
                details={},
            )

        try:
            files = _collect_files(target, glob_pattern)
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {search_path}")],
                details={},
            )

        results: list[str] = []
        count = 0
        is_directory = target.is_dir()

        for file_path in files:
            if signal is not None and signal.is_set():
                break
            if count >= limit:
                break
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, UnicodeDecodeError, IsADirectoryError):
                continue

            lines = content.splitlines()
            for line_no, line in enumerate(lines, 1):
                if signal is not None and signal.is_set():
                    break
                if count >= limit:
                    break
                if not regex.search(line):
                    continue
                rel_path = _format_path(file_path, target, is_directory)
                results.append(f"{rel_path}:{line_no}:{_truncate_line(line)}")
                count += 1
                # context 行（前后 N 行，以 `-` 分隔，对齐 TS rg --context 输出）。
                for ctx_no in range(
                    max(1, line_no - context), min(len(lines), line_no + context) + 1
                ):
                    if ctx_no == line_no:
                        continue
                    results.append(f"{rel_path}:{ctx_no}-{_truncate_line(lines[ctx_no - 1])}")

        if not results:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"No matches found for pattern: {pattern}")],
                details={"matches": 0},
            )

        match_limit_reached = count >= limit
        output = f"Found {count} match(es):\n" + "\n".join(results)
        if match_limit_reached:
            output += f"\n[Truncated: {limit} matches limit]"

        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details={
                "matches": count,
                "matchLimitReached": limit if match_limit_reached else None,
            },
        )

    return AgentTool(
        name="grep",
        prompt_snippet="Search file contents for patterns (respects .gitignore)",
        description=(
            "Search file contents for a pattern under the working directory. "
            "Returns matching lines with file paths and line numbers. "
            "Output is truncated to 100 matches. "
            "Never search from the filesystem root (e.g. grep -r /) or scan the whole disk."
        ),
        input_schema=TOOL_SCHEMA,
        label="Grep",
        execute=execute,
    )


def _format_path(file_path: Path, search_root: Path, is_directory: bool) -> str:
    """对齐 TS formatPath：目录搜索时相对路径，否则文件名。"""
    if is_directory and file_path.is_relative_to(search_root):
        return file_path.relative_to(search_root).as_posix()
    return file_path.name


def _truncate_line(line: str) -> str:
    if len(line) <= GREP_MAX_LINE_LENGTH:
        return line
    return line[:GREP_MAX_LINE_LENGTH] + "…"


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
    ignored = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
    parts = set(path.parts)
    return bool(parts & ignored)
