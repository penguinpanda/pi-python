"""Markdown 渲染（对齐 TS Markdown 组件：标题/列表/代码块/表格/链接）。"""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text


def rich_markdown(text: str) -> Markdown:
    """Rich Markdown 渲染对象。"""
    return Markdown(text, code_theme="monokai")


def render_labeled_markdown(label: str, text: str) -> Any:
    """消息条目渲染：label（粗体）+ Markdown 正文。

    扩展 markdown transformer 已产出 Rich markup（含 [/ 闭合标签）时，
    保留原样文本，避免二次转义。
    """
    label_text = Text(f"{label} ", style="bold")
    if "[/" in text:
        return Group(label_text, text)
    return Group(label_text, rich_markdown(text))
