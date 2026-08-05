"""Overlay Test - demo overlay anchors / borders / clear.

Python port of overlay-test.ts（简化：无内联输入，用 set_overlay 展示）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    shown = {"value": False}

    async def handler(ctx, args: str) -> None:
        shown["value"] = not shown["value"]
        if shown["value"]:
            ctx.ui.set_overlay(
                "overlay-test",
                [
                    "Overlay test",
                    "wide chars: 中文 █ ▓ ▒",
                    "styled: [accent]accent[/accent]",
                    "Press /overlay-test to close",
                ],
                {
                    "anchor": "top-right",
                    "border": "round",
                    "border_color": "accent",
                    "title": "overlay-test",
                },
            )
        else:
            ctx.ui.set_overlay("overlay-test", [])

    pi.register_command(
        "overlay-test",
        {
            "description": "Test overlay rendering with edge cases",
            "handler": handler,
        },
    )
