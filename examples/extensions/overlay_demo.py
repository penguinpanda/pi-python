"""Overlay Demo Extension - absolute-positioned overlay with border and animation.

没有直接对应的 TS 单文件示例；演示 set_overlay 的锚点 / 边框 / 动画。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    shown = {"value": False}

    def handler(ctx, args: str) -> None:
        shown["value"] = not shown["value"]
        if shown["value"]:
            ctx.ui.set_overlay(
                "demo",
                ["Pi overlay demo", "Press /overlay to close"],
                {
                    "anchor": "center",
                    "border": "round",
                    "border_color": "accent",
                    "title": "overlay",
                    "animate": True,
                    "duration": 0.3,
                },
            )
        else:
            ctx.ui.set_overlay("demo", [])

    pi.register_command(
        "overlay",
        {
            "description": "Toggle a demo overlay",
            "handler": handler,
        },
    )
