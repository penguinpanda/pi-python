"""Overlay QA Tests - exercise set_overlay anchors / borders / animation.

Python port of overlay-qa-tests.ts（简化：/overlay-qa 顺序跑一组检查并汇报）。
"""

import asyncio

from pi_coding_agent import ExtensionAPI


ANCHORS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        results: list[str] = []

        # 1) 五个锚点 + 边框/标题
        for anchor in ANCHORS:
            ctx.ui.set_overlay(
                "qa",
                [f"anchor: {anchor}", "wide chars: 中文 █"],
                {"anchor": anchor, "border": "round", "border_color": "accent", "title": "qa"},
            )
            await asyncio.sleep(0.05)
            results.append(f"{anchor}: ok")

        # 2) 动画过渡
        ctx.ui.set_overlay(
            "qa",
            ["animated overlay"],
            {"anchor": "center", "animate": True, "duration": 0.1},
        )
        await asyncio.sleep(0.2)
        results.append("animate: ok")

        # 3) 清空移除
        ctx.ui.set_overlay("qa", [])
        results.append("clear: ok")

        ctx.ui.notify("Overlay QA:\n" + "\n".join(results), "info")

    pi.register_command(
        "overlay-qa",
        {
            "description": "Run overlay rendering QA checks",
            "handler": handler,
        },
    )
