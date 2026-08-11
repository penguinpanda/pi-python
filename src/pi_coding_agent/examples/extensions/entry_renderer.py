"""Custom entry rendering example - append_entry + register_entry_renderer.

Python port of entry-renderer.ts。条目不进入 LLM 上下文，/tree 里用渲染器显示。
"""

import time

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def render_status_card(entry, state):
        data = entry.get("data") or {}
        message = data.get("message", "No data")
        text = f"[status] {message}"
        if state.get("expanded"):
            text += f"\n  {data.get('timestamp', '')}"
        return text

    pi.register_entry_renderer("status-card", render_status_card)

    def status_card_command(ctx, args: str) -> None:
        pi.append_entry(
            "status-card",
            {
                "message": args.strip() or "Status card",
                "timestamp": int(time.time() * 1000),
            },
        )

    pi.register_command(
        "status-card",
        {
            "description": "Render a durable status card that is not sent to the LLM",
            "handler": status_card_command,
        },
    )
