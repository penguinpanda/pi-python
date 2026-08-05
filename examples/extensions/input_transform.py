"""Input Transform Example - the `input` event for intercepting user input.

Python port of input-transform.ts.

Type inside pi:
  ?quick What is X?  -> "Respond briefly: What is X?"
  ping               -> "pong" (instant, no LLM)
  time               -> current time (instant, no LLM)
"""

from datetime import datetime

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def on_input(event, ctx):
        # 跳过扩展注入的消息
        if event.get("source") == "extension":
            return {"action": "continue"}

        text = event.get("text", "")

        # Transform: ?quick prefix for brief responses
        if text.startswith("?quick "):
            query = text[7:].strip()
            if not query:
                ctx.ui.notify("Usage: ?quick <question>", "warning")
                return {"action": "handled"}
            return {
                "action": "transform",
                "text": f"Respond briefly in 1-2 sentences: {query}",
            }

        # Handled: instant responses without LLM
        if text.lower() == "ping":
            ctx.ui.notify("pong", "info")
            return {"action": "handled"}
        if text.lower() == "time":
            ctx.ui.notify(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "info")
            return {"action": "handled"}

        return {"action": "continue"}

    pi.on("input", on_input)
