"""Inter-extension event bus example - communication via pi.events.

Python port of event-bus.ts. TS 的 session_start 事件尚未接线，
这里用 agent_start 缓存 ExtensionContext。

Usage: /emit [message]
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    current_ctx = {"ctx": None}

    async def on_agent_start(event, ctx):
        current_ctx["ctx"] = ctx

    def on_my_notification(data):
        message = data.get("message", "")
        sender = data.get("from", "unknown")
        ctx = current_ctx["ctx"]
        if ctx is not None:
            ctx.ui.notify(f"Event from {sender}: {message}", "info")

    def emit_command(ctx, args: str):
        message = args.strip() or "hello"
        pi.events.emit("my:notification", {"message": message, "from": "/emit command"})

    pi.on("agent_start", on_agent_start)
    pi.events.on("my:notification", on_my_notification)
    pi.register_command(
        "emit",
        {
            "description": "Emit my:notification event (usage: /emit message)",
            "handler": emit_command,
        },
    )
