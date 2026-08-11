"""Custom Header Extension - replace the header text via ctx.ui.set_header.

Python port of custom-header.ts（字符串版）。
"""

from pi_coding_agent import ExtensionAPI


def _decorate(ctx, color: str, text: str) -> str:
    theme = getattr(ctx.ui, "theme", None)
    if theme is not None:
        return theme.fg(color, text)
    return text


def _mascot(ctx) -> str:
    block = _decorate(ctx, "accent", "█")
    pupil = _decorate(ctx, "dim", "▌")
    eye = f"{block}{pupil}"
    return "\n".join(
        [
            "",
            f"     {eye}  {eye}",
            f"  {_decorate(ctx, 'accent', '█' * 14)}",
            *[f"     {block}{block}    {block}{block}" for _ in range(4)],
            _decorate(ctx, "dim", "   shitty coding agent"),
            "",
        ]
    )


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        if ctx.mode == "tui":
            ctx.ui.set_header(_mascot(ctx))

    def restore(ctx, args: str) -> None:
        ctx.ui.set_header(None)
        ctx.ui.notify("Built-in header restored", "info")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "builtin-header",
        {
            "description": "Restore built-in header with keybinding hints",
            "handler": restore,
        },
    )
