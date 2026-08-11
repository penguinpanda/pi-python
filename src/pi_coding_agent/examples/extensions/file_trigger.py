"""File Trigger Extension - poll a trigger file and inject contents.

Python port of file-trigger.ts（fs.watch 改为轮询，跨平台）。
"""

import asyncio
from pathlib import Path

from pi_coding_agent import ExtensionAPI


TRIGGER_FILE = Path("/tmp/agent-trigger.txt")


def create_extension(pi: ExtensionAPI):
    task_holder: dict[str, asyncio.Task | None] = {"task": None}

    async def _poll(ctx) -> None:
        while True:
            try:
                if TRIGGER_FILE.is_file():
                    content = TRIGGER_FILE.read_text(encoding="utf-8").strip()
                    if content:
                        pi.send_message(
                            content,
                            {"customType": "file-trigger", "display": True},
                        )
                        TRIGGER_FILE.write_text("", encoding="utf-8")
            except OSError:
                pass
            await asyncio.sleep(1.0)

    def on_session_start(event, ctx):
        task = asyncio.create_task(_poll(ctx))
        task_holder["task"] = task
        if ctx.has_ui:
            ctx.ui.notify(f"Watching {TRIGGER_FILE}", "info")

    def on_shutdown(event, ctx):
        task = task_holder.get("task")
        if task is not None:
            task.cancel()
            task_holder["task"] = None

    pi.on("session_start", on_session_start)
    pi.on("session_shutdown", on_shutdown)
