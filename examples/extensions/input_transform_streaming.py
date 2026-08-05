"""Streaming-Aware Input Gate - skip expensive pre-processing during steering.

Python port of input-transform-streaming.ts。
"""

import re

from pi_coding_agent import ExtensionAPI


TRIGGER = re.compile(r"\b(changes?|diff|modified)\b", re.IGNORECASE)


def create_extension(pi: ExtensionAPI):
    async def on_input(event, ctx):
        if event.get("streamingBehavior") == "steer":
            return {"action": "continue"}
        text = str(event.get("text", ""))
        if not TRIGGER.search(text):
            return {"action": "continue"}
        try:
            result = await pi.exec("git", ["diff", "--stat"])
        except Exception:
            return {"action": "continue"}
        output = str(result.get("output", "")).strip()
        if result.get("exit_code") != 0 or not output:
            return {"action": "continue"}
        return {
            "action": "transform",
            "text": f"{text}\n\nCurrent uncommitted changes:\n```\n{output}\n```",
        }

    pi.on("input", on_input)
