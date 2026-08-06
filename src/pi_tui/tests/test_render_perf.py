"""大内容量渲染性能回归测试：跨帧缓存 + 增量差分。"""

from __future__ import annotations

from pi_tui.components import MessageEntry
from pi_tui.engine import App, FakeTerminal
from pi_tui.engine.widgets import ScrollView, Vertical


def test_message_entry_caches_rendered_lines_across_frames() -> None:
    entry = MessageEntry("User", "hello")
    first = entry.render(40, 1)
    second = entry.render(40, 1)
    assert first is second
    assert all(a is b for a, b in zip(first, second, strict=True))
    entry.set_text("hello world")
    third = entry.render(40, 1)
    assert third is not first


def test_message_entry_caches_natural_size_across_frames() -> None:
    entry = MessageEntry("User", "word " * 30)
    tall = entry.natural_size(40)[1]
    assert entry.natural_size(40) == entry.natural_size(40)
    entry.set_text("changed")
    short = entry.natural_size(40)[1]
    assert tall > short


def test_regular_mode_rewrites_only_changed_message_rows() -> None:
    """200 条历史消息时，更新末尾一条不应重写其它消息行。"""
    term = FakeTerminal(size=(40, 12))
    app = App(terminal=term, ui_mode="regular")
    body = Vertical()
    entries = []
    for index in range(200):
        entry = MessageEntry("User", f"message {index}")
        body.mount(entry)
        entries.append(entry)
    view = ScrollView(body)
    # regular 模式布局：chat 按内容自然高度展开（basis="auto"）。
    app.screen.mount(view, basis="auto", grow=0, shrink=1, min_size=1)

    app._render_regular(force=True)
    assert "message 0" in term.output_text
    term.reset_output()

    entries[-1].set_text("message 199 updated")
    app._render_regular(False)
    delta = term.output_text
    assert "message 199 updated" in delta
    assert "message 0" not in delta
    assert "message 100" not in delta
    assert "\x1b[2J" not in delta
