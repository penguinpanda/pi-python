"""Timed dialogs example - auto-cancel confirm/select with countdown.

Python port of timed-confirm.ts. pi-python 的 timeout 参数是秒（float），
不是 TS 的毫秒对象。

Commands:
- /timed         - confirm 5 秒自动取消
- /timed-select  - select 10 秒自动取消
- /timed-signal  - asyncio.wait_for 手工超时
"""

import asyncio

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def timed_command(ctx, args: str):
        confirmed = await ctx.ui.confirm(
            "Timed Confirmation",
            "This dialog will auto-cancel in 5 seconds. Confirm?",
            timeout=5.0,
        )
        ctx.ui.notify("Confirmed by user!" if confirmed else "Cancelled or timed out", "info")

    async def timed_select_command(ctx, args: str):
        choice = await ctx.ui.select(
            "Pick an option",
            ["Option A", "Option B", "Option C"],
            timeout=10.0,
        )
        ctx.ui.notify(
            f"Selected: {choice}" if choice else "Selection cancelled or timed out",
            "info",
        )

    async def timed_signal_command(ctx, args: str):
        ctx.ui.notify("Dialog will auto-cancel in 5 seconds...", "info")
        try:
            confirmed = await asyncio.wait_for(
                ctx.ui.confirm(
                    "Timed Confirmation",
                    "This dialog will auto-cancel in 5 seconds. Confirm?",
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            ctx.ui.notify("Dialog timed out (auto-cancelled)", "warning")
            return
        ctx.ui.notify("Confirmed by user!" if confirmed else "Cancelled by user", "info")

    pi.register_command(
        "timed",
        {
            "description": "Show a timed confirmation dialog (auto-cancels in 5s)",
            "handler": timed_command,
        },
    )
    pi.register_command(
        "timed-select",
        {
            "description": "Show a timed select dialog (auto-cancels in 10s)",
            "handler": timed_select_command,
        },
    )
    pi.register_command(
        "timed-signal",
        {
            "description": "Show a timed confirm using asyncio timeout (manual approach)",
            "handler": timed_signal_command,
        },
    )
