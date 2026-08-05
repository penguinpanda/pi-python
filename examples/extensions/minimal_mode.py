"""Minimal Mode - compact tool result rendering.

Python port of minimal-mode.ts：用 register_tool_renderer 只显示工具调用摘要。
"""

from pi_coding_agent import ExtensionAPI


def _shorten_path(path: str) -> str:
    import os

    home = os.path.expanduser("~")
    return path.replace(home, "~", 1) if path.startswith(home) else path


def create_extension(pi: ExtensionAPI):
    def _bash(message):
        command = str(message.get("command", "")) or str(
            (message.get("details") or {}).get("command", "")
        )
        return f"bash: {command[:80]}"

    def _read(message):
        return f"read: {_shorten_path(str((message.get('details') or {}).get('path', '?')))}"

    def _edit(message):
        return f"edit: {_shorten_path(str((message.get('details') or {}).get('path', '?')))}"

    def _write(message):
        return f"write: {_shorten_path(str((message.get('details') or {}).get('path', '?')))}"

    def _grep(message):
        return f"grep: {str((message.get('details') or {}).get('pattern', '?'))[:60]}"

    def _generic(message):
        tool = message.get("tool_name", "tool")
        return f"{tool}: done"

    for tool, renderer in (
        ("bash", _bash),
        ("read", _read),
        ("edit", _edit),
        ("write", _write),
        ("grep", _grep),
    ):
        pi.register_tool_renderer(tool, renderer)
    pi.register_tool_renderer("find", _generic)
    pi.register_tool_renderer("ls", _generic)
