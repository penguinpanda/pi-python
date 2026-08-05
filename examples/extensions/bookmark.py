"""Entry bookmarking example - set_label for /tree navigation.

Python port of bookmark.ts。
"""

import time

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def _entries(ctx):
        if ctx.session is None:
            return []
        return ctx.session.session_manager.get_entries()

    def bookmark_command(ctx, args: str) -> None:
        label = args.strip() or f"bookmark-{int(time.time() * 1000)}"
        for entry in reversed(_entries(ctx)):
            if entry.get("type") == "message":
                message = entry.get("message") or {}
                if message.get("role") == "assistant":
                    pi.set_label(entry["id"], label)
                    ctx.ui.notify(f"Bookmarked as: {label}", "info")
                    return
        ctx.ui.notify("No assistant message to bookmark", "warning")

    def unbookmark_command(ctx, args: str) -> None:
        for entry in reversed(_entries(ctx)):
            if entry.get("type") == "label":
                pi.set_label(entry.get("targetId"), None)
                ctx.ui.notify(f"Removed bookmark: {entry.get('label')}", "info")
                return
        ctx.ui.notify("No bookmarked entry found", "warning")

    pi.register_command(
        "bookmark",
        {
            "description": "Bookmark last message (usage: /bookmark [label])",
            "handler": bookmark_command,
        },
    )
    pi.register_command(
        "unbookmark",
        {
            "description": "Remove bookmark from last labeled entry",
            "handler": unbookmark_command,
        },
    )
