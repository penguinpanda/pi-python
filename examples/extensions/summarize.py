"""Summarize Extension - summarize the conversation via model_registry.

Python port of summarize.ts（简化：结果写入编辑器）。
"""

from pi_coding_agent import ExtensionAPI


def _conversation_text(ctx) -> str:
    if ctx.session is None:
        return ""
    sections: list[str] = []
    for message in ctx.session.get_messages():
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = str(content or "")
        if text.strip():
            label = "User" if role == "user" else "Assistant"
            sections.append(f"{label}: {text.strip()}")
    return "\n\n".join(sections)


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        if ctx.model is None or ctx.session is None:
            ctx.ui.notify("No model selected", "error")
            return
        text = _conversation_text(ctx)
        if not text:
            ctx.ui.notify("No conversation to summarize", "warning")
            return
        from pi_ai import Context as AiContext
        from pi_ai import now_ms

        response = await ctx.model_registry.complete(
            ctx.model,
            AiContext(
                system_prompt="You are a conversation summarizer. Be concise.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Summarize this conversation with Goal / Progress / "
                                    "Key Decisions / Next Steps.\n\n"
                                    f"<conversation>\n{text}\n</conversation>"
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
        summary = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        ctx.ui.set_editor_text(summary)
        ctx.ui.notify("Summary loaded into editor", "info")

    pi.register_command(
        "summarize",
        {
            "description": "Summarize the conversation into the editor",
            "handler": handler,
        },
    )
