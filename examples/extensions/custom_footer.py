"""Custom Footer Extension - toggle custom footer text via ctx.ui.set_footer.

Python port of custom-footer.ts（字符串版；TS 的 renderer 回调简化为文本）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    enabled = {"value": False}

    def handler(ctx, args: str) -> None:
        enabled["value"] = not enabled["value"]
        if enabled["value"]:
            stats = ctx.session.get_session_stats() if ctx.session is not None else {}
            tokens = stats.get("tokens", {}) or {}
            cost = stats.get("cost", 0) or 0

            def fmt(n) -> str:
                return str(n) if n < 1000 else f"{n / 1000:.1f}k"

            input_t = fmt(tokens.get("input", 0) or 0)
            output_t = fmt(tokens.get("output", 0) or 0)
            model = ctx.model.id if ctx.model is not None else "no-model"
            ctx.ui.set_footer(f"up {input_t} down {output_t} ${cost:.3f}  {model}")
            ctx.ui.notify("Custom footer enabled", "info")
        else:
            ctx.ui.set_footer(None)
            ctx.ui.notify("Default footer restored", "info")

    pi.register_command(
        "footer",
        {
            "description": "Toggle custom footer",
            "handler": handler,
        },
    )
