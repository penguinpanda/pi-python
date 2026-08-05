"""Dynamic Tools Extension - register tools after session initialization.

Python port of dynamic-tools.ts。
"""

import re

from pi_coding_agent import ExtensionAPI, ToolDefinition


NAME_RE = re.compile(r"^[a-z0-9_]+$")


def create_extension(pi: ExtensionAPI):
    registered: set[str] = set()

    def register_echo_tool(name: str, label: str, prefix: str) -> bool:
        if name in registered:
            return False
        registered.add(name)

        def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
            return {
                "content": [{"type": "text", "text": f"{prefix}{params['message']}"}],
                "details": {"tool": name, "prefix": prefix},
            }

        pi.register_tool(
            ToolDefinition(
                name=name,
                label=label,
                description=f"Echo a message with prefix: {prefix}",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "Message to echo"}},
                    "required": ["message"],
                },
                execute=execute,
            )
        )
        return True

    def on_session_start(event, ctx):
        register_echo_tool("echo_session", "Echo Session", "[session] ")
        ctx.ui.notify("Registered dynamic tool: echo_session", "info")

    def add_echo_tool(ctx, args: str) -> None:
        name = args.strip().lower()
        if not name or not NAME_RE.match(name):
            ctx.ui.notify(
                "Usage: /add-echo-tool <tool_name> (lowercase, numbers, underscores)",
                "warning",
            )
            return
        created = register_echo_tool(name, f"Echo {name}", f"[{name}] ")
        if not created:
            ctx.ui.notify(f"Tool already registered: {name}", "warning")
            return
        ctx.ui.notify(f"Registered dynamic tool: {name}", "info")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "add-echo-tool",
        {
            "description": "Register a new echo tool dynamically: /add-echo-tool <tool_name>",
            "handler": add_echo_tool,
        },
    )
