"""TruncatedText 组件测试。"""

from __future__ import annotations

from pi_tui.engine import TruncatedText


def test_truncated_text_uses_first_line() -> None:
    widget = TruncatedText("first line\nsecond line")
    lines = widget.render(20, 3)
    assert lines[0].text().startswith("first line")
    assert "second line" not in lines[0].text()


def test_truncated_text_padding_and_height() -> None:
    widget = TruncatedText("x", padding_x=1, padding_y=1)
    lines = widget.render(10, 4)
    assert len(lines) == 4
    assert lines[1].text().startswith(" x")
    assert lines[0].text().strip() == ""
    assert widget.content_size() == (3, 3)
