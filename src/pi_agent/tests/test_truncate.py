"""工具输出截断（truncate.py）单元测试。"""

from __future__ import annotations

from pi_agent.truncate import (
    format_size,
    split_lines_for_counting,
    truncate_head,
    truncate_line,
    truncate_tail,
    utf8_byte_length,
)


def test_utf8_byte_length_and_lines():
    assert utf8_byte_length("héllo") == 6
    assert split_lines_for_counting("") == []
    assert split_lines_for_counting("a\nb") == ["a", "b"]
    assert split_lines_for_counting("a\nb\n") == ["a", "b"]


def test_format_size():
    assert format_size(512) == "512B"
    assert format_size(2048) == "2.0KB"
    assert format_size(2 * 1024 * 1024) == "2.0MB"


def test_truncate_head_no_truncation():
    result = truncate_head("a\nb\nc", max_lines=10, max_bytes=1000)
    assert result.truncated is False
    assert result.content == "a\nb\nc"
    assert result.total_lines == 3


def test_truncate_head_line_limit():
    content = "\n".join(f"line{i}" for i in range(5))
    result = truncate_head(content, max_lines=3, max_bytes=1000)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.content == "line0\nline1\nline2"
    assert result.output_lines == 3


def test_truncate_head_byte_limit_mid_line():
    # 每行 5 字节 + 换行，前两行 11 字节，第三行超 12 字节上限。
    result = truncate_head("aaaaa\nbbbbb\nccccc", max_lines=10, max_bytes=12)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "aaaaa\nbbbbb"


def test_truncate_head_first_line_exceeds_limit():
    result = truncate_head("x" * 100, max_lines=10, max_bytes=50)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == ""
    assert result.first_line_exceeds_limit is True


def test_truncate_head_exact_boundary_not_truncated():
    content = "a\nb"
    result = truncate_head(content, max_lines=2, max_bytes=utf8_byte_length(content))
    assert result.truncated is False


def test_truncate_head_empty():
    result = truncate_head("", max_lines=1, max_bytes=1)
    assert result.truncated is False
    assert result.content == ""


def test_truncate_tail_line_limit_keeps_last_lines():
    content = "\n".join(f"line{i}" for i in range(5))
    result = truncate_tail(content, max_lines=2, max_bytes=1000)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.content == "line3\nline4"


def test_truncate_tail_byte_limit_partial_last_line():
    result = truncate_tail("aaaaa\nbbbbb\nccccc", max_lines=10, max_bytes=3)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert result.content == "ccc"


def test_truncate_tail_empty():
    result = truncate_tail("", max_lines=1, max_bytes=1)
    assert result.truncated is False


def test_truncate_line():
    assert truncate_line("short") == ("short", False)
    truncated, flag = truncate_line("x" * 20, max_chars=5)
    assert flag is True
    assert truncated.startswith("xxxxx")
    assert "[truncated]" in truncated
