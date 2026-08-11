"""Built-in Tool Renderer - compact rendering for read/bash/edit/write.

Python port of built-in-tool-renderer.ts（渲染器层；不改工具行为）。
"""

from pi_coding_agent import ExtensionAPI


def _content_text(message) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def create_extension(pi: ExtensionAPI):
    def read_renderer(message):
        text = _content_text(message)
        line_count = len(text.splitlines()) if text else 0
        details = message.get("details") or {}
        truncated = bool((details.get("truncation") or {}).get("truncated"))
        suffix = " (truncated)" if truncated else ""
        return f"read: {line_count} lines{suffix}"

    def bash_renderer(message):
        details = message.get("details") or {}
        command = str(details.get("command", ""))[:60]
        exit_code = details.get("exitCode")
        return f"bash [{command}]: exit {exit_code if exit_code is not None else '?'}"

    def edit_renderer(message):
        text = _content_text(message)
        return f"edit: {len(text.splitlines())} line(s) changed"

    def write_renderer(message):
        text = _content_text(message)
        return f"write: {len(text.encode('utf-8'))} bytes"

    pi.register_tool_renderer("read", read_renderer)
    pi.register_tool_renderer("bash", bash_renderer)
    pi.register_tool_renderer("edit", edit_renderer)
    pi.register_tool_renderer("write", write_renderer)
