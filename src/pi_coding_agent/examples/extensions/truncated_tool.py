"""Truncated Tool Example - rg tool with proper output truncation.

Python port of truncated-tool.ts（无自定义渲染器；保留头部截断 + 临时文件提示）。
"""

import tempfile
from pathlib import Path

from pi_coding_agent import ExtensionAPI, ToolDefinition


MAX_LINES = 2000
MAX_BYTES = 50 * 1024


def _truncate_head(output: str) -> tuple[str, bool, int, int]:
    lines = output.split("\n")
    total_lines = len(lines)
    text = "\n".join(lines[:MAX_LINES])
    truncated = len(text.encode("utf-8")) > MAX_BYTES
    if truncated:
        text = text.encode("utf-8")[:MAX_BYTES].decode("utf-8", errors="ignore")
    return text, truncated or total_lines > MAX_LINES, total_lines, len(text.encode("utf-8"))


def create_extension(pi: ExtensionAPI):
    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        pattern = str(params.get("pattern", ""))
        search_path = str(params.get("path", "."))
        glob = params.get("glob")
        cwd = ctx.cwd if ctx is not None else "."
        args = ["rg", "--line-number", "--color=never"]
        if glob:
            args += ["--glob", str(glob)]
        args += [pattern, search_path]
        try:
            result = await pi.exec("rg", args, {"timeout": 30, "cwd": cwd})
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"ripgrep failed: {exc}"}],
                "details": {"pattern": pattern, "matchCount": 0},
            }
        output = str(result.get("output", ""))
        if result.get("exit_code") == 1 or not output.strip():
            return {
                "content": [{"type": "text", "text": "No matches found"}],
                "details": {"pattern": pattern, "matchCount": 0},
            }
        text, truncated, total_lines, output_bytes = _truncate_head(output)
        details = {
            "pattern": pattern,
            "matchCount": len([line for line in output.splitlines() if line.strip()]),
        }
        if truncated:
            temp_file = Path(tempfile.mkdtemp(prefix="pi-rg-")) / "output.txt"
            temp_file.write_text(output, encoding="utf-8")
            details["fullOutputPath"] = str(temp_file)
            text += (
                f"\n\n[Output truncated: showing first {MAX_LINES} lines "
                f"({output_bytes} bytes). Full output saved to: {temp_file}]"
            )
        return {
            "content": [{"type": "text", "text": text}],
            "details": details,
        }

    pi.register_tool(
        ToolDefinition(
            name="rg",
            label="ripgrep",
            description=(
                "Search file contents using ripgrep. Output is truncated to "
                f"{MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). "
                "If truncated, full output is saved to a temp file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)"},
                    "path": {"type": "string", "description": "Directory to search"},
                    "glob": {"type": "string", "description": "File glob pattern"},
                },
                "required": ["pattern"],
            },
            execute=execute,
        )
    )
