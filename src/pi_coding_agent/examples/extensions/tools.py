"""Tools Extension - /tools command to enable/disable tools.

Python port of tools.ts（无 SettingsList UI，简化为命令行切换）。
状态通过 append_entry("tools-config") 持久化到会话分支。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    enabled: set[str] = set()

    def restore(ctx) -> None:
        nonlocal enabled
        all_names = {tool["name"] for tool in pi.get_all_tools()}
        saved = None
        if ctx.session is not None:
            for entry in ctx.session.session_manager.get_branch():
                if entry.get("type") == "custom" and entry.get("customType") == "tools-config":
                    data = entry.get("data") or {}
                    if isinstance(data.get("enabledTools"), list):
                        saved = data["enabledTools"]
        if saved is not None:
            enabled = {name for name in saved if name in all_names}
        else:
            enabled = set(pi.get_active_tools())
        pi.set_active_tools(sorted(enabled))

    def handler(ctx, args: str) -> None:
        nonlocal enabled
        parts = args.split()
        all_tools = {tool["name"] for tool in pi.get_all_tools()}
        if parts and parts[0] in ("on", "off") and len(parts) == 2:
            name = parts[1]
            if name not in all_tools:
                ctx.ui.notify(f"Unknown tool: {name}", "warning")
                return
            if parts[0] == "on":
                enabled.add(name)
            else:
                enabled.discard(name)
            pi.set_active_tools(sorted(enabled))
            if ctx.session is not None:
                pi.append_entry("tools-config", {"enabledTools": sorted(enabled)})
            ctx.ui.notify(f"{name} {parts[0]}", "info")
            return
        ctx.ui.notify(
            "Tools:\n"
            + "\n".join(
                f"{'[x]' if name in enabled else '[ ]'} {name}" for name in sorted(all_tools)
            ),
            "info",
        )

    pi.register_command(
        "tools",
        {
            "description": "Enable/disable tools (usage: /tools [on|off <name>])",
            "handler": handler,
        },
    )

    def on_session_start(event, ctx):
        restore(ctx)

    pi.on("session_start", on_session_start)
