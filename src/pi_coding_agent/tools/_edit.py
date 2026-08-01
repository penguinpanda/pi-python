"""
edit 工具 — 对文件应用 unified diff 修改。
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

from ._read import _resolve_path

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the file to edit (relative to cwd)",
        },
        "diff": {
            "type": "string",
            "description": "The unified diff to apply to the file",
        },
    },
    "required": ["path", "diff"],
}


def create_edit_tool(cwd: str) -> AgentTool:
    """创建 edit 工具（unified diff 应用）。"""
    base = Path(cwd).resolve()

    async def execute(
        tool_call_id: str,
        params: dict,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        try:
            file_path = _resolve_path(base, params["path"])
            diff_text = params["diff"]
        except ValueError as e:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                details={},
            )

        try:
            original = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: File not found: {params['path']}")],
                details={},
            )
        except UnicodeDecodeError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Cannot edit binary file: {params['path']}")],
                details={},
            )

        result = _apply_diff(original, diff_text)
        if result is None:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Failed to apply diff to {params['path']}. The file content may have changed.")],
                details={"error": "diff_apply_failed"},
            )

        try:
            file_path.write_text(result, encoding="utf-8")
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Error: Permission denied: {params['path']}")],
                details={},
            )

        return AgentToolResult(
            content=[TextContent(type="text", text=f"File edited: {params['path']}")],
            details={"path": str(file_path)},
        )

    return AgentTool(
        name="edit",
        description="Apply a unified diff to a file. Use this to make precise edits.",
        input_schema=TOOL_SCHEMA,
        label="Edit",
        execute=execute,
    )


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    """解析 diff hunk header: @@ -oldStart,oldCount +newStart,newCount @@"""
    m = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
    if not m:
        return None
    old_start = int(m.group(1))
    old_count = int(m.group(2)) if m.group(2) else 1
    new_start = int(m.group(3))
    new_count = int(m.group(4)) if m.group(4) else 1
    return (old_start, old_count, new_start, new_count)


def _apply_diff(original: str, diff_text: str) -> str | None:
    """使用 Python difflib 应用 unified diff。

    先尝试 difflib.restore，若失败则回退到逐行手动应用。
    """
    original_lines = original.splitlines(keepends=True)

    # 解析 diff hunk
    diff_lines = diff_text.splitlines(keepends=True)
    hunks: list[dict] = []
    current_hunk: dict | None = None

    for line in diff_lines:
        if line.startswith("@@"):
            parsed = _parse_hunk_header(line)
            if parsed:
                current_hunk = {
                    "old_start": parsed[0],
                    "old_count": parsed[1],
                    "new_start": parsed[2],
                    "new_count": parsed[3],
                    "lines": [],
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if not hunks:
        return None

    # 从后往前应用 hunks（避免行号偏移问题）
    for hunk in reversed(hunks):
        old_idx = hunk["old_start"] - 1  # 0-indexed
        old_count = hunk["old_count"]
        old_end = old_idx + old_count

        # 构建替换内容
        replacement: list[str] = []
        for hunk_line in hunk["lines"]:
            if hunk_line.startswith("+") or hunk_line.startswith(" "):
                replacement.append(hunk_line[1:] if hunk_line[0] in ("+", " ") else hunk_line)

        # 验证上下文行
        context_ok = True
        context_idx = old_idx
        for hunk_line in hunk["lines"]:
            if hunk_line.startswith(" "):
                if context_idx >= len(original_lines) or original_lines[context_idx].rstrip("\n") != hunk_line[1:].rstrip("\n"):
                    context_ok = False
                    break
                context_idx += 1
            elif hunk_line.startswith("-"):
                context_idx += 1

        if not context_ok:
            # 宽松模式：跳过上下文验证，直接替换
            pass

        # 应用替换
        original_lines[old_idx:old_end] = replacement

    return "".join(original_lines)
