"""Handoff Extension - generate a focused prompt for a new session.

Python port of handoff.ts（简化：摘要写入编辑器，不创建新会话）。
"""

from pi_coding_agent import ExtensionAPI


SYSTEM_PROMPT = (
    "You are a context transfer assistant. Given a conversation history and the "
    "user's goal for a new thread, generate a focused prompt that:\n"
    "1. Summarizes relevant context (decisions, approaches, key findings)\n"
    "2. Lists relevant files discussed or modified\n"
    "3. Clearly states the next task\n"
    "4. Is self-contained\n"
    "Output only the prompt itself.\n"
)


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("handoff requires interactive mode", "error")
            return
        if ctx.model is None or ctx.session is None:
            ctx.ui.notify("No model selected", "error")
            return
        goal = args.strip()
        if not goal:
            goal_input = await ctx.ui.input("Handoff goal", "What should the new session do?")
            if not goal_input:
                return
            goal = goal_input
        history = "\n".join(
            f"[{message.get('role', '?')}]: {str(message.get('content'))[:400]}"
            for message in ctx.session.get_messages()
        )
        from pi_ai import Context as AiContext
        from pi_ai import now_ms

        response = await ctx.model_registry.complete(
            ctx.model,
            AiContext(
                system_prompt=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Goal: {goal}\n\n<conversation>\n{history}\n</conversation>"
                                ),
                            }
                        ],
                        "timestamp": now_ms(),
                    }
                ],
            ),
            {"max_tokens": 2048},
        )
        blocks = response.get("content") or []
        prompt = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if prompt:
            ctx.ui.set_editor_text(prompt)
            ctx.ui.notify("Handoff prompt loaded into editor", "info")
        else:
            ctx.ui.notify("Failed to generate handoff prompt", "error")

    pi.register_command(
        "handoff",
        {
            "description": "Transfer context to a new focused session",
            "handler": handler,
        },
    )
