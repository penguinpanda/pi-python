"""Tool Override Example - override the built-in read tool.

Python port of tool-override.ts：同名注册覆盖内置 read；记录日志并阻断敏感路径。
"""

import re
import time
from pathlib import Path

from pi_coding_agent import ExtensionAPI, ToolDefinition


BLOCKED_PATTERNS = [
    re.compile(r"\.env$"),
    re.compile(r"\.env\..+$"),
    re.compile(r"secrets?\.(json|yaml|yml|toml)$", re.IGNORECASE),
    re.compile(r"credentials?\.(json|yaml|yml|toml)$", re.IGNORECASE),
]


def create_extension(pi: ExtensionAPI):
    log_file = {"path": None}

    def _log(ctx, path: str, allowed: bool, reason: str = "") -> None:
        if log_file["path"] is None or ctx.session is None:
            return
        status = "ALLOWED" if allowed else "BLOCKED"
        suffix = f" ({reason})" if reason else ""
        try:
            with Path(log_file["path"]).open("a", encoding="utf-8") as handle:
                handle.write(f"[{time.time():.0f}] {status}: {path}{suffix}\n")
        except OSError:
            pass

    def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        path = str(params.get("path", ""))
        cwd = ctx.cwd if ctx is not None else "."
        absolute = str((Path(cwd) / path).resolve())
        if any(pattern.search(absolute) for pattern in BLOCKED_PATTERNS):
            _log(ctx, absolute, False, "matches blocked pattern")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f'Access denied: "{path}" matches a blocked pattern (sensitive file).',
                    }
                ],
                "details": {"blocked": True},
            }
        _log(ctx, absolute, True)
        try:
            lines = Path(absolute).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return {
                "content": [{"type": "text", "text": f"Error reading {path}: {exc}"}],
                "details": {"blocked": False},
            }
        start = max(0, int(params.get("offset", 1)) - 1)
        limit = params.get("limit")
        end = start + int(limit) if limit is not None else len(lines)
        text = "\n".join(lines[start:end])
        return {
            "content": [{"type": "text", "text": text or "(empty)"}],
            "details": {"blocked": False},
        }

    def on_session_start(event, ctx):
        if ctx.session is not None:
            from pi_coding_agent._config import get_agent_dir

            log_file["path"] = str(get_agent_dir() / "read-access.log")

    pi.register_tool(
        ToolDefinition(
            name="read",
            label="read (audited)",
            description="Read the contents of a file with access logging. Some sensitive paths are blocked.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from",
                    },
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
            execute=execute,
        )
    )
    pi.on("session_start", on_session_start)
