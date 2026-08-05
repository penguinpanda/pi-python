"""Snake Extension - play snake in a set_overlay board.

Python port of snake.ts（简化：方向用 /snake-up|down|left|right 命令控制，
覆盖层由 set_overlay 渲染，支持会话内重开）。
"""

import asyncio
import random

from pi_coding_agent import ExtensionAPI


WIDTH = 20
HEIGHT = 10
TICK_S = 0.25


def _board_lines(snake: list[tuple[int, int]], food: tuple[int, int]) -> list[str]:
    cells = set(snake)
    lines = []
    for y in range(HEIGHT):
        lines.append(
            "".join(
                "*" if (x, y) == food else ("█" if (x, y) in cells else ".") for x in range(WIDTH)
            )
        )
    return lines


def create_extension(pi: ExtensionAPI):
    game = {
        "running": False,
        "task": None,
        "snake": [(5, 5), (4, 5), (3, 5)],
        "direction": (1, 0),
        "food": (10, 5),
        "score": 0,
    }

    def _spawn_food():
        while True:
            pos = (random.randrange(WIDTH), random.randrange(HEIGHT))
            if pos not in game["snake"]:
                return pos

    def _redraw(ctx) -> None:
        ctx.ui.set_overlay(
            "snake",
            [*_board_lines(game["snake"], game["food"]), f"Score: {game['score']}"],
            {"anchor": "center", "border": "round", "title": "snake"},
        )

    async def _tick(ctx) -> None:
        while game["running"]:
            head = game["snake"][0]
            dx, dy = game["direction"]
            new_head = (head[0] + dx, head[1] + dy)
            hit_wall = not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT)
            if hit_wall or new_head in game["snake"]:
                game["running"] = False
                ctx.ui.set_overlay(
                    "snake",
                    ["GAME OVER", f"Score: {game['score']}", "Run /snake to play again"],
                    {"anchor": "center", "border": "round", "title": "snake"},
                )
                return
            game["snake"].insert(0, new_head)
            if new_head == game["food"]:
                game["score"] += 1
                game["food"] = _spawn_food()
            else:
                game["snake"].pop()
            _redraw(ctx)
            await asyncio.sleep(TICK_S)

    def _set_direction(ctx, args: str, direction: tuple[int, int]) -> None:
        game["direction"] = direction

    async def start(ctx, args: str) -> None:
        if game["running"]:
            ctx.ui.notify("Snake already running; use /snake-stop first", "info")
            return
        game.update(
            running=True,
            snake=[(5, 5), (4, 5), (3, 5)],
            direction=(1, 0),
            food=(10, 5),
            score=0,
        )
        _redraw(ctx)
        game["task"] = asyncio.create_task(_tick(ctx))
        ctx.ui.notify("Snake started. Use /snake-up|down|left|right to steer.", "info")

    async def stop(ctx, args: str) -> None:
        game["running"] = False
        task = game.get("task")
        if task is not None:
            task.cancel()
            game["task"] = None
        ctx.ui.set_overlay("snake", [])
        ctx.ui.notify("Snake stopped", "info")

    pi.register_command(
        "snake",
        {
            "description": "Start a snake game in an overlay",
            "handler": start,
        },
    )
    pi.register_command(
        "snake-stop",
        {
            "description": "Stop the snake game",
            "handler": stop,
        },
    )
    for name, direction in (
        ("snake-up", (0, -1)),
        ("snake-down", (0, 1)),
        ("snake-left", (-1, 0)),
        ("snake-right", (1, 0)),
    ):
        pi.register_command(
            name,
            {
                "description": f"Steer snake {name.split('-')[-1]}",
                "handler": lambda ctx, args, d=direction: _set_direction(ctx, args, d),
            },
        )
