"""Dirty Repo Guard - prevent session changes with uncommitted git changes.

Python port of dirty-repo-guard.ts。
"""

from pi_coding_agent import ExtensionAPI


async def _check_dirty(pi: ExtensionAPI, ctx, action: str):
    try:
        result = await pi.exec("git", ["status", "--porcelain"])
    except Exception:
        return None
    output = str(result.get("output", ""))
    if result.get("exit_code") != 0 or not output.strip():
        return None
    if not ctx.has_ui:
        return {"cancel": True}
    changed = len([line for line in output.splitlines() if line.strip()])
    choice = await ctx.ui.select(
        f"You have {changed} uncommitted file(s). {action} anyway?",
        ["Yes, proceed anyway", "No, let me commit first"],
    )
    if choice != "Yes, proceed anyway":
        ctx.ui.notify("Commit your changes first", "warning")
        return {"cancel": True}
    return None


def create_extension(pi: ExtensionAPI):
    async def on_before_switch(event, ctx):
        action = "new session" if event.get("position") == "before" else "switch session"
        return await _check_dirty(pi, ctx, action)

    async def on_before_fork(event, ctx):
        return await _check_dirty(pi, ctx, "fork")

    pi.on("session_before_switch", on_before_switch)
    pi.on("session_before_fork", on_before_fork)
