"""GitHub Issue Autocomplete - register_autocomplete provider.

Python port of github-issue-autocomplete.ts（简化：session_start 预载 gh issues）。
"""

import asyncio
import json

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    issues = {"items": []}

    async def _load(ctx) -> None:
        try:
            result = await pi.exec(
                "gh",
                ["issue", "list", "--limit", "20", "--json", "number,title"],
            )
        except Exception:
            return
        if result.get("exit_code") != 0:
            return
        try:
            data = json.loads(str(result.get("output", "")))
        except json.JSONDecodeError:
            return
        if isinstance(data, list):
            issues["items"] = data

    def on_session_start(event, ctx):
        asyncio.create_task(_load(ctx))

    def provider(text: str):
        if "#" not in text:
            return None
        return [
            {
                "value": f"#{item.get('number')}",
                "label": f"#{item.get('number')} {str(item.get('title', ''))[:40]}",
            }
            for item in issues["items"]
        ]

    pi.register_autocomplete(provider)
    pi.on("session_start", on_session_start)
