"""Hidden Thinking Label Extension - set_hidden_thinking_label.

Python port of hidden-thinking-label.ts。
"""

from pi_coding_agent import ExtensionAPI


DEFAULT_LABEL = "Pondering..."


def create_extension(pi: ExtensionAPI):
    label = {"value": DEFAULT_LABEL}

    def apply(ctx) -> None:
        ctx.ui.set_hidden_thinking_label(label["value"])

    def on_session_start(event, ctx):
        apply(ctx)

    def handler(ctx, args: str) -> None:
        next_label = args.strip()
        if not next_label:
            label["value"] = DEFAULT_LABEL
            ctx.ui.set_hidden_thinking_label()
            ctx.ui.notify(f"Hidden thinking label reset to: {DEFAULT_LABEL}")
            return
        label["value"] = next_label
        ctx.ui.set_hidden_thinking_label(next_label)
        ctx.ui.notify(f"Hidden thinking label set to: {next_label}")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "thinking-label",
        {
            "description": "Set the hidden thinking label. Use without args to reset.",
            "handler": handler,
        },
    )
