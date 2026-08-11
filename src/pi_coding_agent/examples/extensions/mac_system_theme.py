"""macOS system theme sync - set_theme with dark/light detection.

Python port of mac-system-theme.ts（仅 darwin 上运行 osascript，其余平台提示不支持）。
"""

import asyncio
import sys

from pi_coding_agent import ExtensionAPI


async def _is_dark_mode() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import subprocess

        completed = await asyncio.to_thread(
            subprocess.run,
            [
                "osascript",
                "-e",
                'tell application "System Events" to tell appearance preferences to return dark mode',
            ],
            capture_output=True,
            text=True,
        )
        return (completed.stdout or "").strip() == "true"
    except Exception:
        return False


def create_extension(pi: ExtensionAPI):
    holder: dict[str, asyncio.Task | None] = {"task": None}

    async def _watch(ctx) -> None:
        current = "dark" if await _is_dark_mode() else "light"
        ctx.ui.set_theme(current)
        while True:
            await asyncio.sleep(2.0)
            new_theme = "dark" if await _is_dark_mode() else "light"
            if new_theme != current:
                current = new_theme
                ctx.ui.set_theme(current)

    def on_session_start(event, ctx):
        if sys.platform != "darwin":
            ctx.ui.notify("mac-system-theme requires macOS", "warning")
            return
        holder["task"] = asyncio.create_task(_watch(ctx))

    def on_shutdown(event, ctx):
        task = holder.get("task")
        if task is not None:
            task.cancel()
            holder["task"] = None

    pi.on("session_start", on_session_start)
    pi.on("session_shutdown", on_shutdown)
