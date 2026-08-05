"""Plan Mode - toggle read-only exploration tools.

Python port of plan-mode/（简化：/plan 只读工具，/implement 恢复全部工具）。
"""

from pi_coding_agent import ExtensionAPI


READONLY_TOOLS = ["read", "grep", "find", "ls"]


def create_extension(pi: ExtensionAPI):
    state = {"plan": False, "all_tools": []}

    def on_session_start(event, ctx):
        if ctx.session is not None:
            state["all_tools"] = [tool.name for tool in ctx.session._agent.state.tools]

    async def handler(ctx, args: str) -> None:
        if not state["all_tools"]:
            state["all_tools"] = [tool["name"] for tool in pi.get_all_tools()]
        if args.strip() == "on":
            state["plan"] = True
        elif args.strip() == "off":
            state["plan"] = False
        else:
            state["plan"] = not state["plan"]
        if state["plan"]:
            pi.set_active_tools([name for name in READONLY_TOOLS if name in state["all_tools"]])
            ctx.ui.notify(
                "PLANNING MODE: read-only tools enabled. No edits or writes allowed.",
                "info",
            )
        else:
            pi.set_active_tools(list(state["all_tools"]))
            ctx.ui.notify("Implementation mode restored", "info")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "plan",
        {
            "description": "Toggle read-only planning mode (/plan [on|off])",
            "handler": handler,
        },
    )
