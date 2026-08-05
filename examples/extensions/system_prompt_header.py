"""System Prompt Header - show system prompt length in status.

Python port of system-prompt-header.ts。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def on_agent_start(event, ctx):
        prompt = ctx.get_system_prompt()
        ctx.ui.set_status("system-prompt", f"System: {len(prompt)} chars")

    def on_shutdown(event, ctx):
        ctx.ui.set_status("system-prompt", None)

    pi.on("agent_start", on_agent_start)
    pi.on("session_shutdown", on_shutdown)
