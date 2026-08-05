"""Interactive Shell Extension - user_bash operations override.

Python port of interactive-shell.ts（简化：operations.exec 用 subprocess 执行）。
"""

import asyncio

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def _exec(command: str, cwd: str, options=None):
        import subprocess

        options = options or {}
        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                ),
                timeout=float(options.get("timeout") or 120),
            )
            return {
                "output": (completed.stdout or "") + (completed.stderr or ""),
                "exitCode": completed.returncode,
                "cancelled": False,
                "truncated": False,
            }
        except asyncio.TimeoutError:
            return {
                "output": "[timed out]",
                "exitCode": 124,
                "cancelled": False,
                "truncated": False,
            }

    async def on_user_bash(event, ctx):
        return {"operations": {"exec": _exec}}

    pi.on("user_bash", on_user_bash)
