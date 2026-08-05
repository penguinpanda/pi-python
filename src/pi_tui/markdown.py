"""Markdown 渲染（对齐 TS Markdown 组件：标题/列表/代码块/表格/链接）。"""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


def rich_markdown(text: str) -> Markdown:
    """Rich Markdown 渲染对象。"""
    return Markdown(text, code_theme="monokai")


def label_icon(label: str) -> str:
    """按消息标签返回图标（User/Assistant/Tool/Bash/System 等）。"""
    lowered = label.lower()
    if lowered.startswith("user"):
        return "👤"
    if lowered.startswith("assistant"):
        return "🤖"
    if lowered.startswith("tool:"):
        return "🛠"
    if lowered.startswith("tool call"):
        return "🔧"
    if lowered.startswith("bash"):
        return "$"
    if lowered.startswith("thinking"):
        return "💭"
    if lowered.startswith("skill"):
        return "✨"
    if lowered.startswith("system"):
        return "⚙️"
    if lowered.startswith("compaction"):
        return "📦"
    if lowered.startswith("branch"):
        return "🌿"
    return "▸"


def render_labeled_markdown(label: str, text: str, *, speaking: bool = False) -> Any:
    """消息条目渲染：图标 + label（粗体，单独一行）+ 缩进正文。

    扩展 markdown transformer 已产出 Rich markup（含 [/ 闭合标签）时，
    保留原样文本，避免二次转义。
    """
    suffix = " Speaking…" if speaking else ""
    label_text = Text(f"{label_icon(label)} {label}{suffix}", style="bold")
    body: Any
    if "[/" in text:
        body = text
    else:
        body = rich_markdown(text)
    return Group(label_text, Padding(body, (0, 0, 0, 2)))
