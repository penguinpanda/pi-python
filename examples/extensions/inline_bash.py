"""Inline Bash Extension - expand !{command} patterns in prompts.

Python port of inline-bash.ts。
"""

import re

from pi_coding_agent import ExtensionAPI


PATTERN = re.compile(r"!\{([^}]+)\}")


def create_extension(pi: ExtensionAPI):
    async def on_input(event, ctx):
        text = str(event.get("text", ""))
        stripped = text.lstrip()
        if stripped.startswith("!") and not stripped.startswith("!{"):
            return {"action": "continue"}
        if not PATTERN.search(text):
            return {"action": "continue"}

        matches = list(PATTERN.finditer(text))
        result = text
        expansions: list[str] = []
        for match in matches:
            command = match.group(1)
            try:
                bash_result = await pi.exec("bash", ["-c", command], {"timeout": 30})
                output = str(bash_result.get("output", ""))
                trimmed = output.strip()
                if bash_result.get("exit_code") != 0 and "stderr" not in bash_result:
                    trimmed = f"[error: exit code {bash_result.get('exit_code')}]"
            except Exception as exc:
                trimmed = f"[error: {exc}]"
            result = result.replace(match.group(0), trimmed)
            expansions.append(f"!{{{command}}} -> {trimmed[:50]}")

        if ctx.has_ui and expansions:
            ctx.ui.notify("Expanded inline command(s):\n" + "\n".join(expansions), "info")
        return {"action": "transform", "text": result}

    pi.on("input", on_input)
