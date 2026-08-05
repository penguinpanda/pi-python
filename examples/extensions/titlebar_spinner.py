"""Titlebar Spinner Extension - braille spinner in terminal title while working.

Python port of titlebar-spinner.ts。
"""

import asyncio
from pathlib import Path

from pi_coding_agent import ExtensionAPI


BRAILLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def create_extension(pi: ExtensionAPI):
    holder: dict[str, asyncio.Task | None] = {"task": None}

    def base_title(pi) -> str:
        cwd = Path.cwd().name
        session = pi.get_session_name()
        return f"π - {session} - {cwd}" if session else f"π - {cwd}"

    async def _spin(ctx) -> None:
        frame = 0
        while True:
            title = f"{BRAILLE_FRAMES[frame % len(BRAILLE_FRAMES)]} {base_title(pi)}"
            ctx.ui.set_title(title)
            frame += 1
            await asyncio.sleep(0.08)

    def on_agent_start(event, ctx):
        task = asyncio.create_task(_spin(ctx))
        holder["task"] = task

    def stop(ctx) -> None:
        task = holder.get("task")
        if task is not None:
            task.cancel()
            holder["task"] = None
        ctx.ui.set_title(base_title(pi))

    pi.on("agent_start", on_agent_start)
    pi.on("agent_end", stop)
    pi.on("session_shutdown", stop)
