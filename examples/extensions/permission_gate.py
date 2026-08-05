"""Permission Gate Extension - confirm dangerous bash commands.

Python port of permission-gate.ts. Uses the `tool_call` event
(pi-python 已实现：handler 返回 {"block": True, "reason": ...} 阻断工具)。
"""

import re

from pi_coding_agent import ExtensionAPI


DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+(-rf?|--recursive)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\b(chmod|chown)\b.*777", re.IGNORECASE),
]


def create_extension(pi: ExtensionAPI):
    async def on_tool_call(event, ctx):
        if event.get("toolName") != "bash":
            return None

        command = str((event.get("input") or {}).get("command", ""))
        is_dangerous = any(pattern.search(command) for pattern in DANGEROUS_PATTERNS)
        if not is_dangerous:
            return None

        if not ctx.has_ui:
            # 非交互模式默认阻断
            return {
                "block": True,
                "reason": "Dangerous command blocked (no UI for confirmation)",
            }

        choice = await ctx.ui.select(
            f"Dangerous command:\n\n  {command}\n\nAllow?",
            ["Yes", "No"],
        )
        if choice != "Yes":
            return {"block": True, "reason": "Blocked by user"}
        return None

    pi.on("tool_call", on_tool_call)
