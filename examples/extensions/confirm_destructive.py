"""Confirm Destructive Actions Extension - cancel session replacement with confirm.

Python port of confirm-destructive.ts（事件用 position 区分）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def on_before_switch(event, ctx):
        if not ctx.has_ui:
            return None
        if event.get("position") == "before":
            confirmed = await ctx.ui.confirm(
                "Clear session?",
                "This will delete all messages in the current session.",
            )
            if not confirmed:
                ctx.ui.notify("Clear cancelled", "info")
                return {"cancel": True}
            return None
        # position == "at"（switch）
        entries = []
        if ctx.session is not None:
            entries = ctx.session.session_manager.get_entries()
        has_work = any(
            entry.get("type") == "message" and (entry.get("message") or {}).get("role") == "user"
            for entry in entries
        )
        if has_work:
            confirmed = await ctx.ui.confirm(
                "Switch session?",
                "You have messages in the current session. Switch anyway?",
            )
            if not confirmed:
                ctx.ui.notify("Switch cancelled", "info")
                return {"cancel": True}
        return None

    async def on_before_fork(event, ctx):
        if not ctx.has_ui:
            return None
        entry_id = str(event.get("entryId", ""))[:8]
        choice = await ctx.ui.select(
            f"Fork from entry {entry_id}?",
            ["Yes, create fork", "No, stay in current session"],
        )
        if choice != "Yes, create fork":
            ctx.ui.notify("Fork cancelled", "info")
            return {"cancel": True}
        return None

    pi.on("session_before_switch", on_before_switch)
    pi.on("session_before_fork", on_before_fork)
