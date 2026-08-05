"""Prompt Customizer Extension - add tool guidance to the system prompt.

Python port of prompt-customizer.ts. 使用 `before_agent_start` 事件：
handler 返回 {"system_prompt": ...} 覆盖当轮系统提示（结束后恢复）。
"""

from pi_coding_agent import ExtensionAPI


def _add_tool_guidance(tool_names: list[str], skill_names: list[str], base_prompt: str) -> str:
    parts: list[str] = []
    if "read" in tool_names:
        parts.append(
            "• Use the `read` tool for file contents (supports text and images).\n"
            "  - For large files, use `offset` and `limit` to read in chunks."
        )
    if "bash" in tool_names:
        parts.append(
            "• Execute commands with the `bash` tool. Use it for file operations "
            "like `ls`, `find`, `grep`."
        )
    if "edit" in tool_names:
        parts.append(
            "• Use the `edit` tool for precise text replacements in files. "
            "Match exact content including whitespace."
        )
    if "write" in tool_names:
        parts.append(
            "• Use the `write` tool to create new files or overwrite existing ones completely."
        )
    if skill_names:
        parts.append(f"\nAvailable skills: {', '.join(skill_names)}")
        parts.append("Use skill documentation for best practices on specific tools.")
    if not parts:
        return base_prompt
    return f"{base_prompt}\n\n## Tool Guidance\n\n{chr(10).join(parts)}\n"


def _merge_with_user_append(base_prompt: str) -> str:
    extension_specific = (
        "\n## Extension-Added Context\n\n"
        "This prompt includes tool guidance and skill information loaded dynamically. "
        "If you have additional requirements, configure them via --append-system-prompt "
        "or project context files.\n"
    )
    return f"{base_prompt}{extension_specific}"


def create_extension(pi: ExtensionAPI):
    async def on_before_agent_start(event, ctx):
        session = ctx.session
        tool_names = [
            tool.name for tool in (session._agent.state.tools if session is not None else [])
        ]
        skill_names = []
        if session is not None and session.skill_loader is not None:
            skill_names = [skill.name for skill in session.skill_loader.all()]
        custom = _add_tool_guidance(tool_names, skill_names, event["system_prompt"])
        return {"system_prompt": _merge_with_user_append(custom)}

    pi.on("before_agent_start", on_before_agent_start)
