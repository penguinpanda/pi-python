"""Pirate Extension - toggle pirate mode via before_agent_start.

Python port of pirate.ts。
"""

from pi_coding_agent import ExtensionAPI


PIRATE_APPEND = """

IMPORTANT: You are now in PIRATE MODE. You must:
- Speak like a stereotypical pirate in all responses
- Use phrases like "Arrr!", "Ahoy!", "Shiver me timbers!", "Avast!", "Ye scurvy dog!"
- Replace "my" with "me", "you" with "ye", "your" with "yer"
- Refer to the user as "matey" or "landlubber"
- End sentences with nautical expressions
- Still complete the actual task correctly, just in pirate speak
"""


def create_extension(pi: ExtensionAPI):
    pirate_mode = {"value": False}

    def toggle(ctx, args: str) -> None:
        pirate_mode["value"] = not pirate_mode["value"]
        ctx.ui.notify(
            "Arrr! Pirate mode enabled!" if pirate_mode["value"] else "Pirate mode disabled",
            "info",
        )

    async def on_before_agent_start(event, ctx):
        if pirate_mode["value"]:
            return {"system_prompt": event["system_prompt"] + PIRATE_APPEND}
        return None

    pi.register_command(
        "pirate",
        {
            "description": "Toggle pirate mode (agent speaks like a pirate)",
            "handler": toggle,
        },
    )
    pi.on("before_agent_start", on_before_agent_start)
