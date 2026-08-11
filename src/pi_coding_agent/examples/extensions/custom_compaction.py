"""Custom Compaction Extension - replace the default compaction summary.

Python port of custom-compaction.ts. 演示 session_before_compact：
返回 {"compaction": {...}} 完全替代内置摘要。

优先用 ctx.model_registry 选择模型生成摘要（TS 版用 Gemini Flash），
找不到模型或调用失败时回退启发式摘要。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    async def on_before_compact(event, ctx):
        ctx.ui.notify("Custom compaction extension triggered", "info")
        preparation = event.get("preparation")
        if preparation is None:
            return None

        messages = list(preparation.messages_to_summarize) + list(preparation.turn_prefix_messages)
        compaction = {
            "firstKeptEntryId": preparation.first_kept_entry_id,
            "tokensBefore": preparation.tokens_before,
        }

        # 1) 模型摘要：用当前会话模型（或 registry.find 指定模型）。
        registry = ctx.model_registry
        model = None
        if registry is not None and ctx.model is not None:
            model = registry.find(ctx.model.provider, ctx.model.id)
        if model is not None and messages:
            try:
                from pi_ai import Context as AiContext
                from pi_ai import now_ms

                conversation_text = "\n".join(
                    f"[{message.get('role', '?')}]: {str(message.get('content'))[:500]}"
                    for message in messages
                )
                prompt_text = (
                    "Create a comprehensive structured summary of this conversation "
                    "with Goal / Progress / Key Decisions / Next Steps / Critical Context. "
                    f"\n\n<conversation>\n{conversation_text}\n</conversation>"
                )
                response = await registry.complete(
                    model,
                    AiContext(
                        system_prompt="You are a conversation summarizer.",
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text}],
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
                ).strip()
                if summary:
                    compaction["summary"] = summary
                    compaction["usage"] = response.get("usage")
                    return {"compaction": compaction}
            except Exception as exc:
                ctx.ui.notify(f"Model summary failed, using heuristic: {exc}", "warning")

        # 2) 启发式回退。
        roles: dict[str, int] = {}
        for message in messages:
            role = message.get("role", "?")
            roles[role] = roles.get(role, 0) + 1

        compaction["summary"] = (
            "## Goal\nContinue the task from the compacted history.\n\n"
            "## Progress\n"
            f"- Compacted {len(messages)} messages "
            f"({preparation.tokens_before} tokens before)\n"
            f"- Roles: {roles}\n\n"
            "## Next Steps\n- Re-read key files and continue the current work.\n"
        )
        return {"compaction": compaction}

    pi.on("session_before_compact", on_before_compact)
