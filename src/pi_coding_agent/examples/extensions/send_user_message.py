"""Send User Message Example - pi.send_user_message() from extensions.

Python port of send-user-message.ts.
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def _ask(ctx, args: str) -> None:
        if not args.strip():
            ctx.ui.notify("Usage: /ask <message>", "warning")
            return
        if not ctx.is_idle():
            ctx.ui.notify("Agent is busy. Use /steer or /followup instead.", "warning")
            return
        pi.send_user_message(args)

    def _steer(ctx, args: str) -> None:
        if not args.strip():
            ctx.ui.notify("Usage: /steer <message>", "warning")
            return
        # deliverAs=steer 的中断语义由会话端决定；当前统一走 send_user_message。
        pi.send_user_message(args, {"deliverAs": "steer"} if not ctx.is_idle() else None)

    def _followup(ctx, args: str) -> None:
        if not args.strip():
            ctx.ui.notify("Usage: /followup <message>", "warning")
            return
        if ctx.is_idle():
            pi.send_user_message(args)
        else:
            pi.send_user_message(args, {"deliverAs": "followUp"})
            ctx.ui.notify("Follow-up queued", "info")

    def _ask_with(ctx, args: str) -> None:
        if not args.strip():
            ctx.ui.notify("Usage: /askwith <message>", "warning")
            return
        if not ctx.is_idle():
            ctx.ui.notify("Agent is busy", "warning")
            return
        pi.send_user_message(
            [
                {"type": "text", "text": f"User request: {args}"},
                {"type": "text", "text": "Please respond concisely."},
            ]
        )

    pi.register_command(
        "ask",
        {
            "description": "Send a user message to the agent",
            "handler": _ask,
        },
    )
    pi.register_command(
        "steer",
        {
            "description": "Send a steering message (interrupts current processing)",
            "handler": _steer,
        },
    )
    pi.register_command(
        "followup",
        {
            "description": "Queue a follow-up message (waits for current processing)",
            "handler": _followup,
        },
    )
    pi.register_command(
        "askwith",
        {
            "description": "Send a user message with structured content",
            "handler": _ask_with,
        },
    )
