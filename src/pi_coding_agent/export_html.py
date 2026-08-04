"""HTML 会话导出（对齐 TS core/export-html/）。

会话数据 Base64 内嵌 + 服务端渲染（Pygments 高亮代码块）。
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from ._session_manager import SessionManager

_CODE_FENCE_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def collect_session_data(
    session: SessionManager,
    *,
    system_prompt: str | None = None,
    tools: list | None = None,
) -> dict[str, Any]:
    """收集会话数据（header、leafId、分支条目、系统提示、工具）。"""
    return {
        "sessionId": session.session_id,
        "cwd": session.cwd,
        "sessionName": session.session_name,
        "leafId": session.get_leaf_id(),
        "entries": session.get_entries(),
        "systemPrompt": system_prompt,
        "tools": tools or [],
    }


def _highlight_code(code: str, language: str | None) -> str:
    try:
        lexer = get_lexer_by_name(language) if language else guess_lexer(code)
    except ClassNotFound:
        lexer = None
    if lexer is None:
        return html.escape(code)
    return highlight(code, lexer, HtmlFormatter(nowrap=True))


def _render_markdown_with_code(text: str) -> str:
    """渲染文本：转义 + 代码块 Pygments 高亮。"""
    escaped = html.escape(text)

    def _replace(match: re.Match[str]) -> str:
        language = match.group(1) or None
        code = html.unescape(match.group(2))
        return '<pre class="code-block">' + _highlight_code(code, language) + "</pre>"

    return _CODE_FENCE_RE.sub(_replace, escaped)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "thinking":
            parts.append(f"<thinking>{block.get('thinking', '')}</thinking>")
        elif block_type == "toolCall":
            parts.append(
                f'<tool_call name="{block.get("name", "")}">{block.get("raw_arguments", "")}</tool_call>'
            )
        elif block_type == "image":
            parts.append("[image]")
    return "\n".join(parts)


def _render_entries(entries: list[dict[str, Any]]) -> str:
    """把分支条目渲染为 HTML。"""
    blocks: list[str] = []
    for entry in entries:
        entry_type = entry.get("type")
        if entry_type == "message":
            message = entry.get("message") or {}
            role = message.get("role", "agent")
            text = _content_to_text(message.get("content"))
            label = {
                "user": "User",
                "assistant": "Assistant",
                "toolResult": "Tool result",
                "system": "System",
                "compactionSummary": "Compaction",
            }.get(role, role)
            blocks.append(
                f'<div class="entry {role}"><span class="role">{html.escape(label)}</span>'
                f'<div class="body">{_render_markdown_with_code(text)}</div></div>'
            )
        elif entry_type == "compaction":
            blocks.append(
                f'<div class="entry compaction"><span class="role">Compaction</span>'
                f'<div class="body">{html.escape(entry.get("summary", ""))}</div></div>'
            )
        elif entry_type == "branch_summary":
            blocks.append(
                f'<div class="entry branch-summary"><span class="role">Branch summary</span>'
                f'<div class="body">{html.escape(entry.get("summary", ""))}</div></div>'
            )
    return "\n".join(blocks)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>pi session {session_id}</title>
<style>
  body {{ background: {bg}; color: {text}; font-family: ui-monospace, monospace; margin: 0; padding: 2rem; }}
  h1 {{ color: {accent}; font-size: 1.2rem; }}
  .meta {{ color: {dim}; margin-bottom: 1.5rem; }}
  .entry {{ margin-bottom: 1rem; }}
  .role {{ font-weight: bold; color: {accent}; margin-right: 0.5rem; }}
  .entry.user .body {{ border-left: 3px solid {accent}; padding-left: 0.6rem; }}
  .entry.assistant .body {{ border-left: 3px solid {success}; padding-left: 0.6rem; }}
  .entry.toolResult .body {{ border-left: 3px solid {warning}; padding-left: 0.6rem; color: {dim}; }}
  .code-block {{ background: {bg_alt}; padding: 0.6rem; overflow-x: auto; }}
  #session-data {{ display: none; }}
</style>
</head>
<body>
<h1>pi session</h1>
<div class="meta">{session_id} · {cwd} · {entry_count} entries</div>
{entries_html}
<script id="session-data" type="text/plain">{data_b64}</script>
</body>
</html>
"""


def export_session_to_html(
    session: SessionManager,
    output_path: str | Path,
    *,
    theme=None,
    system_prompt: str | None = None,
    tools: list | None = None,
) -> Path:
    """将会话导出为独立 HTML 文件。"""
    data = collect_session_data(
        session,
        system_prompt=system_prompt,
        tools=tools,
    )
    encoded = base64.b64encode(
        json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    ).decode("ascii")

    if theme is not None:
        colors = theme.colors
    else:
        from pi_tui.theme import DARK_THEME

        colors = DARK_THEME
    entries_html = _render_entries(session.get_branch())
    html_content = _HTML_TEMPLATE.format(
        session_id=html.escape(session.session_id),
        cwd=html.escape(session.cwd),
        entry_count=len(session.get_entries()),
        entries_html=entries_html,
        data_b64=encoded,
        bg=colors["bg"],
        bg_alt=colors["bgAlt"],
        text=colors["text"],
        dim=colors["textDim"],
        accent=colors["accent"],
        success=colors["success"],
        warning=colors["warning"],
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")
    return output


__all__ = [
    "collect_session_data",
    "export_session_to_html",
]
