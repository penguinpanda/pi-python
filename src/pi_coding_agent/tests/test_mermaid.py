"""Mermaid 渲染与 transformer 测试。"""

from __future__ import annotations

from pi_coding_agent.modes.interactive.mermaid import create_mermaid_markdown_transformer
from pi_tui.mermaid import render_mermaid

FLOWCHART = """flowchart TD
    A[Start] --> B{Check}
    B -->|yes| C[OK]
    B -->|no| D[Fail]
"""

SEQUENCE = """sequenceDiagram
    participant A as Alice
    participant B as Bob
    A->>B: Hello
    B-->>A: Hi
"""


def test_flowchart_td_renders() -> None:
    art = render_mermaid(FLOWCHART)
    assert art is not None
    assert art.plain
    assert "Start" in "\n".join(art.plain)
    assert "Check" in "\n".join(art.plain)
    assert art.width > 0


def test_flowchart_lr_renders() -> None:
    art = render_mermaid("flowchart LR\n    A --> B\n")
    assert art is not None
    assert "A" in "\n".join(art.plain)
    assert "▶" in "\n".join(art.plain)


def test_sequence_renders() -> None:
    art = render_mermaid(SEQUENCE)
    assert art is not None
    text = "\n".join(art.plain)
    assert "Alice" in text
    assert "Bob" in text
    assert "Hello" in text


def test_unsupported_type_returns_none() -> None:
    assert render_mermaid('pie\n    "A": 1\n') is None
    assert render_mermaid("gantt\n    section S\n") is None


def test_transformer_final_mode_replaces_block() -> None:
    transformer = create_mermaid_markdown_transformer(lambda: "final")
    markdown = "before\n\n```mermaid\nflowchart TD\n    A --> B\n```\n\nafter"
    result = transformer(markdown, {"messageType": "assistant", "isStreaming": False})
    assert "```mermaid" not in result
    assert "A" in result
    assert "before" in result and "after" in result


def test_transformer_off_mode_keeps_block() -> None:
    transformer = create_mermaid_markdown_transformer(lambda: "off")
    markdown = "```mermaid\nflowchart TD\n    A --> B\n```"
    result = transformer(markdown, {"messageType": "assistant", "isStreaming": False})
    assert "```mermaid" in result


def test_transformer_streaming_gated_by_mode() -> None:
    markdown = "```mermaid\nflowchart TD\n    A --> B\n```"
    final_only = create_mermaid_markdown_transformer(lambda: "final")
    assert "```mermaid" in final_only(markdown, {"messageType": "assistant", "isStreaming": True})
    streaming = create_mermaid_markdown_transformer(lambda: "streaming")
    assert "```mermaid" not in streaming(
        markdown, {"messageType": "assistant", "isStreaming": True}
    )


def test_transformer_skips_thinking() -> None:
    transformer = create_mermaid_markdown_transformer(lambda: "streaming")
    markdown = "```mermaid\nflowchart TD\n    A --> B\n```"
    result = transformer(markdown, {"messageType": "assistant-thinking", "isStreaming": True})
    assert "```mermaid" in result


def test_transformer_keeps_block_when_too_wide() -> None:
    transformer = create_mermaid_markdown_transformer(lambda: "final", lambda: 4)
    markdown = "```mermaid\nflowchart TD\n    A --> B\n```"
    result = transformer(markdown, {"messageType": "assistant", "isStreaming": False})
    assert "```mermaid" in result


def test_transformer_warning_path() -> None:
    """非流式渲染带警告：保留原块 + 警告提示（含 theme_fg 着色与多条计数）。"""
    markdown = "```mermaid\nflowchart TD\n    ???bad line???\n    A --> B\n```"

    def _theme_fg(kind, text):
        assert kind == "warning"
        return f"[[{text}]]"

    transformer = create_mermaid_markdown_transformer(
        lambda: "final", lambda: 80, theme_fg=_theme_fg
    )
    result = transformer(markdown, {"messageType": "assistant", "isStreaming": False})
    assert "```mermaid" in result  # 原块保留
    assert "Mermaid diagram not rendered" in result
    assert "[[" in result  # theme_fg 着色

    # 流式下警告不展示（直接渲染）
    streaming = create_mermaid_markdown_transformer(lambda: "streaming")
    result_stream = streaming(markdown, {"messageType": "assistant", "isStreaming": True})
    assert "Mermaid diagram not rendered" not in result_stream
