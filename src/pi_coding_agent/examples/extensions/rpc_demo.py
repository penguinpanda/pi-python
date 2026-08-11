"""RPC Extension UI Demo - exercise all RPC-supported UI methods.

Python port of rpc-demo.ts（无 editor() 方法；命令逐个演示 select/confirm/input）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    turn_count = {"value": 0}

    def on_session_start(event, ctx):
        ctx.ui.set_title("pi RPC Demo")
        ctx.ui.set_widget("rpc-demo", ["--- RPC Extension UI Demo ---", "Loaded and ready."])
        ctx.ui.set_status("rpc-demo", f"Turns: {turn_count['value']}")

    def on_turn_start(event, ctx):
        turn_count["value"] += 1
        ctx.ui.set_status("rpc-demo", f"Turn {turn_count['value']} running...")

    def on_turn_end(event, ctx):
        ctx.ui.set_status("rpc-demo", f"Turn {turn_count['value']} done")

    async def rpc_select(ctx, args: str) -> None:
        choice = await ctx.ui.select("Pick", ["A", "B", "C"])
        ctx.ui.notify(f"Selected: {choice}", "info")

    async def rpc_confirm(ctx, args: str) -> None:
        confirmed = await ctx.ui.confirm("Confirm?", "Are you sure?")
        ctx.ui.notify("Confirmed" if confirmed else "Cancelled", "info")

    async def rpc_input(ctx, args: str) -> None:
        value = await ctx.ui.input("Enter text", args or "placeholder")
        ctx.ui.notify(f"Input: {value}", "info")

    async def rpc_prefill(ctx, args: str) -> None:
        ctx.ui.set_editor_text(args or "prefilled")
        ctx.ui.notify("Editor prefilled", "info")

    pi.on("session_start", on_session_start)
    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
    pi.register_command("rpc-select", {"description": "Demo ui.select", "handler": rpc_select})
    pi.register_command("rpc-confirm", {"description": "Demo ui.confirm", "handler": rpc_confirm})
    pi.register_command("rpc-input", {"description": "Demo ui.input", "handler": rpc_input})
    pi.register_command(
        "rpc-prefill", {"description": "Demo set_editor_text", "handler": rpc_prefill}
    )
