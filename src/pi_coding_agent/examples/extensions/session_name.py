"""Session naming example - /session-name to set or show session name.

Python port of session-name.ts.
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def handler(ctx, args: str):
        name = args.strip()
        if name:
            pi.set_session_name(name)
            ctx.ui.notify(f"Session named: {name}", "info")
        else:
            current = pi.get_session_name()
            ctx.ui.notify(
                f"Session: {current}" if current else "No session name set",
                "info",
            )

    pi.register_command(
        "session-name",
        {
            "description": "Set or show session name (usage: /session-name [new name])",
            "handler": handler,
        },
    )
