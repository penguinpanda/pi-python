"""partial_json 手写部分解析器测试。"""

import pytest

from pi_ai.utils.partial_json import partial_json


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        # 完整输入直接解析
        ('{"a": 1}', {"a": 1}),
        ("[1, 2, 3]", [1, 2, 3]),
        ("null", None),
        ("true", True),
        ("42", 42),
        # 空/空白
        ("", {}),
        ("   ", {}),
        (None, {}),
        # 未闭合对象
        ('{"a": 1', {"a": 1}),
        ('{"a": 1, "b": 2', {"a": 1, "b": 2}),
        ('{"a": 1,', {"a": 1}),
        # 未闭合数组
        ("[1, 2", [1, 2]),
        ("[1,", [1]),
        ("[", []),
        ("{", {}),
        # 未闭合字符串值
        ('{"a": "hel', {"a": "hel"}),
        ('{"a": "', {"a": ""}),
        # 不完整数字 / 字面量
        ('{"a": 1.', {"a": 1}),
        ('{"a": -', {}),
        ('{"a": tru', {}),
        # 嵌套容器
        ('{"a": [1, 2], "b": {"c": "d"', {"a": [1, 2], "b": {"c": "d"}}),
        ('{"a": [1,', {"a": [1]}),
        # 不完整 key：退回到最后一个完整值
        ('{"a": 1, "b', {"a": 1}),
        # 转义内容不截断
        ('{"a": "x\\", y"', {"a": 'x", y'}),
    ],
)
def test_partial_json(partial, expected):
    assert partial_json(partial) == expected


def test_partial_json_ignores_trailing_garbage():
    assert partial_json('{"a": 1} garbage') == {"a": 1}


def test_partial_json_never_raises():
    for text in ["", "not json", "{{{{", '{"a":', "tru", '{"a": tru']:
        assert partial_json(text) == {}
