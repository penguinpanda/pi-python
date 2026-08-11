"""Auto-Commit on Exit Extension - commit changes on session_shutdown.

Python port of auto-commit-on-exit.ts。
"""

from pi_coding_agent import ExtensionAPI


async def _last_assistant_text(ctx) -> str:
    if ctx.session is None:
        return ""
    return ctx.session.get_last_assistant_text() or ""


def create_extension(pi: ExtensionAPI):
    async def on_shutdown(event, ctx):
        try:
            status = await pi.exec("git", ["status", "--porcelain"])
        except Exception:
            return
        if status.get("exit_code") != 0 or not str(status.get("output", "")).strip():
            return
        first_line = (await _last_assistant_text(ctx)).splitlines()
        message = first_line[0] if first_line else "Work in progress"
        commit_message = f"[pi] {message[:50]}{'...' if len(message) > 50 else ''}"
        try:
            await pi.exec("git", ["add", "-A"])
            commit = await pi.exec("git", ["commit", "-m", commit_message])
        except Exception:
            return
        if commit.get("exit_code") == 0 and ctx.has_ui:
            ctx.ui.notify(f"Auto-committed: {commit_message}", "info")

    pi.on("session_shutdown", on_shutdown)
