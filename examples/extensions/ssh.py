"""SSH Remote Execution - delegate user bash to a remote machine.

Python port of ssh.ts（简化：user_bash operations 走 ssh；host 用 PI_SSH_HOST）。
"""

import asyncio
import os

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def _exec(command: str, cwd: str, options=None):
        import subprocess

        remote = os.environ.get("PI_SSH_HOST")
        if not remote:
            return {
                "output": "PI_SSH_HOST not set",
                "exitCode": 1,
                "cancelled": False,
                "truncated": False,
            }
        full_command = f"cd {cwd} && {command}" if cwd else command
        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["ssh", remote, full_command],
                    capture_output=True,
                    text=True,
                ),
                timeout=float((options or {}).get("timeout") or 120),
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
