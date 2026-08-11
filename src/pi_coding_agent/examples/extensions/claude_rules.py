"""Claude Rules Extension - list .claude/rules/*.md in the system prompt.

Python port of claude-rules.ts。
"""

from pathlib import Path

from pi_coding_agent import ExtensionAPI


def _find_markdown_files(directory: Path, base: str = "") -> list[str]:
    results: list[str] = []
    if not directory.is_dir():
        return results
    for entry in sorted(directory.iterdir()):
        relative = f"{base}/{entry.name}" if base else entry.name
        if entry.is_dir():
            results.extend(_find_markdown_files(entry, relative))
        elif entry.is_file() and entry.name.endswith(".md"):
            results.append(relative)
    return results


def create_extension(pi: ExtensionAPI):
    state = {"rules_dir": "", "rule_files": []}

    def on_session_start(event, ctx):
        state["rules_dir"] = str(Path(ctx.cwd) / ".claude" / "rules")
        state["rule_files"] = _find_markdown_files(Path(state["rules_dir"]))
        if state["rule_files"]:
            ctx.ui.notify(
                f"Found {len(state['rule_files'])} rule(s) in .claude/rules/",
                "info",
            )

    async def on_before_agent_start(event, ctx):
        if not state["rule_files"]:
            return None
        rules_list = "\n".join(f"- .claude/rules/{f}" for f in state["rule_files"])
        return {
            "system_prompt": event["system_prompt"]
            + (
                "\n\n## Project Rules\n\nThe following project rules are available "
                f"in .claude/rules/:\n\n{rules_list}\n\n"
                "When working on tasks related to these rules, use the read tool "
                "to load the relevant rule files for guidance.\n"
            )
        }

    pi.on("session_start", on_session_start)
    pi.on("before_agent_start", on_before_agent_start)
