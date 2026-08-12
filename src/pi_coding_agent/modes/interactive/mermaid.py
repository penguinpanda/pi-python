"""Mermaid markdown transformer（对齐 TS components/mermaid.ts）。

把顶层 ```mermaid 代码块替换为 Unicode 终端图；三模式：
off（不渲染）/ final（仅最终消息）/ streaming（流式也渲染）。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pi_tui.mermaid import render_mermaid

_MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def _code_span(line: str) -> str:
    """每行包成行内代码，保留空格与制表字符（对齐 TS codeSpan）。"""
    content = line or "\u00a0"
    longest_run = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if content.startswith("`") or content.endswith("`") else ""
    return f"{fence}{padding}{content}{padding}{fence}"


def create_mermaid_markdown_transformer(
    get_mode: Callable[[], str],
    get_width: Callable[[], int] | None = None,
    theme_fg=None,
) -> Callable[[str, dict[str, Any]], str]:
    """构造 mermaid transformer（对齐 TS createMermaidMarkdownTransformer）。"""

    def _width() -> int:
        if get_width is None:
            return 80
        try:
            value = get_width()
        except Exception:
            return 80
        return int(value) if isinstance(value, (int, float)) else 80

    def transformer(markdown: str, context: dict[str, Any]) -> str:
        mode = get_mode()
        if mode == "off" or context.get("messageType") == "assistant-thinking":
            return markdown
        if context.get("isStreaming") and mode != "streaming":
            return markdown
        available_width = _width()

        def _replace(match: re.Match[str]) -> str:
            code = match.group(1)
            art = render_mermaid(code)
            if art is None or art.width > available_width:
                return match.group(0)
            if not context.get("isStreaming") and art.warnings:
                suffix = f" (+{len(art.warnings) - 1} more)" if len(art.warnings) > 1 else ""
                warning = f"Mermaid diagram not rendered: {art.warnings[0]}{suffix}"
                styled_warning = theme_fg("warning", warning) if theme_fg else warning
                return f"{match.group(0)}\n{_code_span(styled_warning)}  \n"
            return "\n".join(_code_span(line) for line in art.plain) + "\n"

        return _MERMAID_BLOCK_RE.sub(_replace, markdown)

    return transformer
