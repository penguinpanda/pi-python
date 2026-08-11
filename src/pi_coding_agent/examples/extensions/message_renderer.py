"""Custom message rendering example - register_message_renderer + send_message.

Python port of message-renderer.ts。渲染器返回字符串（我们的 TUI 渲染管线）。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def render_status(message):
        details = message.get("details") or {}
        level = details.get("level", "info")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return f"[{level.upper()}] {content}"

    pi.register_message_renderer("status-update", render_status)

    def status_command(ctx, args: str) -> None:
        import time

        parts = args.split()
        level = "info"
        content = args.strip()
        if parts and parts[0] in ("warn", "error"):
            level = parts[0]
            content = " ".join(parts[1:]) or "Status update"
        pi.send_message(
            content,
            {
                "customType": "status-update",
                "display": True,
                "details": {"level": level, "timestamp": int(time.time() * 1000)},
            },
        )

    pi.register_command(
        "status",
        {
            "description": "Send a status message (usage: /status [warn|error] message)",
            "handler": status_command,
        },
    )
