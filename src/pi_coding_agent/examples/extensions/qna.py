"""Q&A extraction extension - extract questions into the editor.

Python port of qna.ts：用 ctx.model_registry 抽取最后一条 assistant 消息中的
问题，结果写入编辑器。
"""

from pi_coding_agent import ExtensionAPI


SYSTEM_PROMPT = (
    "You are a question extractor. Given text from a conversation, extract any "
    "questions that need answering and format them for the user to fill in.\n\n"
    "Output format:\n"
    "- List each question on its own line, prefixed with 'Q: '\n"
    "- After each question, add a blank line for the answer prefixed with 'A: '\n"
    "- If no questions are found, output 'No questions found in the last message.'\n"
)


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("qna requires interactive mode", "error")
            return
        if ctx.model is None or ctx.session is None:
            ctx.ui.notify("No model selected", "error")
            return
        last_text = ctx.session.get_last_assistant_text()
        if not last_text:
            ctx.ui.notify("No assistant messages found", "error")
            return

        from pi_ai import Context as AiContext
        from pi_ai import now_ms

        response = await ctx.model_registry.complete(
            ctx.model,
            AiContext(
                system_prompt=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": last_text}],
                        "timestamp": now_ms(),
                    }
                ],
            ),
            {},
        )
        blocks = response.get("content") or []
        result = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not result.strip():
            ctx.ui.notify("Cancelled", "info")
            return
        ctx.ui.set_editor_text(result)
        ctx.ui.notify("Questions loaded. Edit and submit when ready.", "info")

    pi.register_command(
        "qna",
        {
            "description": "Extract questions from last assistant message into editor",
            "handler": handler,
        },
    )
