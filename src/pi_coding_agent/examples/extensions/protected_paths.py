"""Protected Paths Extension - block write/edit on sensitive paths.

Python port of protected-paths.ts。
"""

from pi_coding_agent import ExtensionAPI


PROTECTED_PATHS = [".env", ".git/", "node_modules/"]


def create_extension(pi: ExtensionAPI):
    async def on_tool_call(event, ctx):
        if event.get("toolName") not in ("write", "edit"):
            return None
        path = str((event.get("input") or {}).get("path", ""))
        if any(protected in path for protected in PROTECTED_PATHS):
            if ctx.has_ui:
                ctx.ui.notify(f"Blocked write to protected path: {path}", "warning")
            return {"block": True, "reason": f'Path "{path}" is protected'}
        return None

    pi.on("tool_call", on_tool_call)
