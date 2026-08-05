"""Project Trust Extension - demonstrates the project_trust event.

Python port of project-trust.ts. Install globally:
  cp project_trust.py ~/.pi/agent/extensions/

Try it in a project containing .pi, AGENTS.md/CLAUDE.md, or .agents/skills.
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    load_count = 0
    load_count += 1

    # 多个 handler 允许；第一个返回 trusted=yes/no 的获胜并抑制内置信任弹窗。
    # 返回 undecided 让其他 handler 或内置流程决定。
    async def on_project_trust(event, ctx):
        ctx.ui.notify(
            f"project_trust fired for {event['cwd']} (mode: {ctx.mode}, load: {load_count})",
            "info",
        )

        choice = await ctx.ui.select(
            f"Project trust for:\n{event['cwd']}",
            [
                "Trust and remember",
                "Trust with note and remember",
                "Trust this session",
                "Do not trust this session",
                "Let built-in prompt decide",
            ],
        )
        if choice == "Trust with note and remember":
            note = await ctx.ui.input("Project trust note", "Optional note for this demo")
            ctx.ui.notify(
                f"Recorded demo note: {note}" if note else "No demo note entered",
                "info",
            )
            return {"trusted": "yes", "remember": True}
        if choice == "Trust and remember":
            return {"trusted": "yes", "remember": True}
        if choice == "Trust this session":
            return {"trusted": "yes"}
        if choice == "Do not trust this session":
            return {"trusted": "no"}
        return {"trusted": "undecided"}

    pi.on("project_trust", on_project_trust)
