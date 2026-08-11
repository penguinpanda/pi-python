"""Space Invaders Extension - play in a set_overlay board.

Python port of space-invaders.ts（简化：方向/开火用命令控制，覆盖层渲染）。
"""

import asyncio

from pi_coding_agent import ExtensionAPI


WIDTH = 24
HEIGHT = 12


def create_extension(pi: ExtensionAPI):
    state = {
        "running": False,
        "task": None,
        "player": WIDTH // 2,
        "invaders": [(x * 2, 1) for x in range(6)],
        "direction": 1,
        "bullets": [],
        "score": 0,
    }

    def _redraw(ctx) -> None:
        cells = set(state["invaders"])
        lines = []
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                if (x, y) in cells:
                    row.append("V")
                elif any(bx == x and by == y for bx, by in state["bullets"]):
                    row.append("|")
                elif y == HEIGHT - 1 and x == state["player"]:
                    row.append("A")
                else:
                    row.append(".")
            lines.append("".join(row))
        lines.append(f"Score: {state['score']}")
        ctx.ui.set_overlay(
            "invaders",
            lines,
            {"anchor": "center", "border": "round", "title": "space invaders"},
        )

    async def _tick(ctx) -> None:
        while state["running"]:
            # 入侵者移动
            dx, dy = state["direction"], 0
            if any(x + dx >= WIDTH or x + dx < 0 for x, _ in state["invaders"]):
                state["direction"] = -state["direction"]
                dx, dy = state["direction"], 1
            moved = [(x + dx, y + dy) for x, y in state["invaders"]]
            if any(y >= HEIGHT - 1 for _, y in moved):
                state["running"] = False
                ctx.ui.set_overlay(
                    "invaders",
                    ["GAME OVER", f"Score: {state['score']}", "Run /invaders to play again"],
                    {"anchor": "center", "border": "round", "title": "space invaders"},
                )
                return
            state["invaders"] = moved
            # 子弹移动与碰撞
            new_bullets = []
            for bx, by in state["bullets"]:
                ny = by - 1
                if any(bx == ix and ny == iy for ix, iy in state["invaders"]):
                    state["invaders"] = [
                        i for i in state["invaders"] if not (i[0] == bx and i[1] == ny)
                    ]
                    state["score"] += 10
                elif ny >= 0:
                    new_bullets.append((bx, ny))
            state["bullets"] = new_bullets
            _redraw(ctx)
            await asyncio.sleep(0.3)

    def _move(ctx, args: str, delta: int) -> None:
        if state["running"]:
            state["player"] = max(0, min(WIDTH - 1, state["player"] + delta))
            _redraw(ctx)

    def _fire(ctx, args: str) -> None:
        if state["running"]:
            state["bullets"].append((state["player"], HEIGHT - 2))

    async def start(ctx, args: str) -> None:
        if state["running"]:
            ctx.ui.notify("Invaders already running; use /invaders-stop first", "info")
            return
        state.update(
            running=True,
            player=WIDTH // 2,
            invaders=[(x * 2, 1) for x in range(6)],
            direction=1,
            bullets=[],
            score=0,
        )
        _redraw(ctx)
        state["task"] = asyncio.create_task(_tick(ctx))
        ctx.ui.notify("Invaders started. Use /invaders-left|right|fire.", "info")

    async def stop(ctx, args: str) -> None:
        state["running"] = False
        task = state.get("task")
        if task is not None:
            task.cancel()
            state["task"] = None
        ctx.ui.set_overlay("invaders", [])
        ctx.ui.notify("Invaders stopped", "info")

    pi.register_command(
        "invaders",
        {
            "description": "Start space invaders in an overlay",
            "handler": start,
        },
    )
    pi.register_command(
        "invaders-stop",
        {
            "description": "Stop the invaders game",
            "handler": stop,
        },
    )
    pi.register_command(
        "invaders-left",
        {
            "description": "Move player left",
            "handler": lambda ctx, args: _move(ctx, args, -1),
        },
    )
    pi.register_command(
        "invaders-right",
        {
            "description": "Move player right",
            "handler": lambda ctx, args: _move(ctx, args, 1),
        },
    )
    pi.register_command(
        "invaders-fire",
        {
            "description": "Fire a bullet",
            "handler": _fire,
        },
    )
