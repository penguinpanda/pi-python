"""Trigger compaction example.

Python port of trigger-compact.ts. 注意：pi-python 的
ExtensionContext.get_context_usage() 目前返回 None（stub），
因此基于 token 阈值的自动触发不可用；只保留 /trigger-compact 手动命令。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def trigger_compact_command(ctx, args: str):
        ctx.ui.notify("Compaction started", "info")
        try:
            await ctx.compact()
        except Exception as exc:
            ctx.ui.notify(f"Compaction failed: {exc}", "error")
            return
        ctx.ui.notify("Compaction completed", "info")

    pi.register_command(
        "trigger-compact",
        {
            "description": "Trigger compaction immediately",
            "handler": trigger_compact_command,
        },
    )
