"""Working Indicator Extension - status-based streaming indicator.

Python port of working-indicator.ts（无独立 indicator 组件，用 set_status 实现）。
"""

from pi_coding_agent import ExtensionAPI


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def create_extension(pi: ExtensionAPI):
    state = {"mode": "spinner", "frame": 0}

    def indicator_text() -> str:
        mode = state["mode"]
        if mode == "dot":
            return "●"
        if mode == "none":
            return ""
        if mode == "pulse":
            return ["·", "•", "●", "•"][state["frame"] % 4]
        return SPINNER_FRAMES[state["frame"] % len(SPINNER_FRAMES)]

    def on_turn_start(event, ctx):
        state["frame"] = 0
        ctx.ui.set_status("working-indicator", indicator_text())

    def on_turn_end(event, ctx):
        ctx.ui.set_status("working-indicator", None)

    def handler(ctx, args: str) -> None:
        mode = args.strip().lower()
        if not mode:
            ctx.ui.notify(f"Working indicator: {state['mode']}", "info")
            return
        if mode == "reset":
            state["mode"] = "spinner"
        elif mode in ("dot", "none", "pulse", "spinner"):
            state["mode"] = mode
        else:
            ctx.ui.notify("Usage: /working-indicator [dot|pulse|none|spinner|reset]", "error")
            return
        ctx.ui.notify(f"Working indicator set to: {state['mode']}", "info")

    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
    pi.register_command(
        "working-indicator",
        {
            "description": "Set the streaming working indicator: dot, pulse, none, spinner, or reset",
            "handler": handler,
        },
    )
