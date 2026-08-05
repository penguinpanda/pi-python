"""Status Line Extension - persistent status text via ctx.ui.set_status.

Python port of status-line.ts. TUI 模式 ctx.ui.theme 可用（fg/bg 颜色），
print 模式回退纯文本。
"""

from pi_coding_agent import ExtensionAPI


def _decorate(ctx, color: str, text: str) -> str:
    theme = getattr(ctx.ui, "theme", None)
    if theme is not None:
        return theme.fg(color, text)
    return text


def create_extension(pi: ExtensionAPI):
    turn_count = 0

    async def on_turn_start(event, ctx):
        nonlocal turn_count
        turn_count += 1
        ctx.ui.set_status("status-demo", _decorate(ctx, "accent", f"● Turn {turn_count}..."))

    async def on_turn_end(event, ctx):
        ctx.ui.set_status(
            "status-demo",
            _decorate(ctx, "success", f"✓ Turn {turn_count} complete"),
        )

    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
