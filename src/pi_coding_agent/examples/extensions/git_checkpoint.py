"""Git Checkpoint Extension - stash checkpoints per turn, restore on fork.

Python port of git-checkpoint.ts。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    checkpoints: dict[str, str] = {}
    current_entry_id = {"value": None}

    async def on_tool_result(event, ctx):
        if ctx.session is not None:
            leaf = ctx.session.session_manager.get_leaf_id()
            current_entry_id["value"] = leaf

    async def on_turn_start(event, ctx):
        try:
            result = await pi.exec("git", ["stash", "create"])
        except Exception:
            return
        ref = str(result.get("output", "")).strip()
        if ref and current_entry_id["value"]:
            checkpoints[current_entry_id["value"]] = ref

    async def on_before_fork(event, ctx):
        ref = checkpoints.get(str(event.get("entryId", "")))
        if not ref or not ctx.has_ui:
            return None
        choice = await ctx.ui.select(
            "Restore code state?",
            ["Yes, restore code to that point", "No, keep current code"],
        )
        if choice and choice.startswith("Yes"):
            try:
                await pi.exec("git", ["stash", "apply", ref])
            except Exception:
                return None
            ctx.ui.notify("Code restored to checkpoint", "info")
        return None

    async def on_agent_end(event, ctx):
        checkpoints.clear()

    pi.on("tool_result", on_tool_result)
    pi.on("turn_start", on_turn_start)
    pi.on("session_before_fork", on_before_fork)
    pi.on("agent_end", on_agent_end)
