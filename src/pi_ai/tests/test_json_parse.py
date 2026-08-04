"""repair_json / parse_streaming_json 测试（对齐 TS json-parse.ts）。"""

import pytest

from pi_ai.utils.json_parse import (
    parse_json_with_repair,
    parse_streaming_json,
    repair_json,
)


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        # 合法转义保持不变
        (r'{"a": "\n\t\"\\"}', r'{"a": "\n\t\"\\"}'),
        # 原始控制字符转义
        ('{"a": "line1\nline2"}', r'{"a": "line1\nline2"}'),
        ('{"a": "tab\there"}', r'{"a": "tab\there"}'),
        ('{"a": "bell\x07"}', r'{"a": "bell\u0007"}'),
        # 非法转义反斜杠加倍
        (r'{"a": "\x"}', r'{"a": "\\x"}'),
        (r'{"a": "\q"}', r'{"a": "\\q"}'),
        # 尾部孤立反斜杠加倍
        ('{"a": "oops\\', '{"a": "oops\\\\'),
        # 合法 \uXXXX 保持不变
        (r'{"a": "\u4e2d"}', r'{"a": "\u4e2d"}'),
        # 非法 \u 序列（不足 4 位十六进制）：TS 行为是保持 \u 不变
        (r'{"a": "\u12"}', r'{"a": "\u12"}'),
    ],
)
def test_repair_json(raw, repaired):
    assert repair_json(raw) == repaired


def test_parse_json_with_repair_recovers_control_chars():
    assert parse_json_with_repair('{"a": "x\ny"}') == {"a": "x\ny"}


def test_parse_json_with_repair_raises_when_unrepairable():
    with pytest.raises(ValueError):
        parse_json_with_repair("not json")


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        (None, {}),
        ("", {}),
        ("   ", {}),
        ('{"a": 1}', {"a": 1}),
        ('{"a": 1, "b": 2', {"a": 1, "b": 2}),
        ('{"a": "hel', {"a": "hel"}),
        ("[1, 2", [1, 2]),
        # 完全无法解析 → {}
        ("not json", {}),
        ('{"a": tru', {}),
    ],
)
def test_parse_streaming_json(partial, expected):
    assert parse_streaming_json(partial) == expected


def test_parse_streaming_json_never_raises():
    for text in [None, "", "}", "{", '{"a":', "tru", "123"]:
        parse_streaming_json(text)  # 不抛异常即可
