"""Commands Extension - list available slash commands via pi.get_commands().

Python port of commands.ts（简化：/commands [extension|prompt|skill]）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        commands = pi.get_commands()
        source_filter = args.strip()
        if source_filter:
            filtered = [
                c
                for c in commands
                if c.source_info and c.source_info.get("source") == source_filter
            ]
        else:
            filtered = commands
        if not filtered:
            ctx.ui.notify(
                f"No {source_filter} commands found" if source_filter else "No commands found",
                "info",
            )
            return
        lines = [f"/{command.name} - {command.description}" for command in filtered]
        selected = await ctx.ui.select("Available Commands", lines)
        if selected:
            ctx.ui.notify(selected, "info")

    pi.register_command(
        "commands",
        {
            "description": "List available slash commands",
            "handler": handler,
        },
    )
