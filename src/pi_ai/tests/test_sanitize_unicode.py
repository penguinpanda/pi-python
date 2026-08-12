"""overflow / sanitize-unicode 补充函数测试。"""

from __future__ import annotations

from pi_ai.utils.overflow import is_recoverable_length
from pi_ai.utils.sanitize_unicode import sanitize_surrogates


def test_is_recoverable_length_matrix() -> None:
    message = {
        "role": "assistant",
        "content": [],
        "stop_reason": "length",
        "usage": {"input": 10, "output": 50, "total_tokens": 60},
    }
    # length + output 低于预期上限 → 可恢复
    assert is_recoverable_length(message, 100) is True
    # 达到上限 → 不可恢复
    assert is_recoverable_length(message, 50) is False
    # 非 length → 不可恢复
    message["stop_reason"] = "stop"
    assert is_recoverable_length(message, 100) is False
    # desired_max_output <= 0 → 不可恢复
    message["stop_reason"] = "length"
    assert is_recoverable_length(message, 0) is False


def test_sanitize_surrogates_removes_unpaired_only() -> None:
    # 正确配对的代理（emoji）保留
    assert sanitize_surrogates("Hello \U0001f648 World") == "Hello \U0001f648 World"
    # 孤立高代理被删除
    assert sanitize_surrogates("Text \ud83d here") == "Text  here"
    # 孤立低代理被删除
    assert sanitize_surrogates("Text \ude00 here") == "Text  here"
    # 混合场景（配对代理以两个码点形式保留，与输入表示一致）
    text = "\ud83d" + "ok" + "\ud83d\ude00" + "\ude00"
    assert sanitize_surrogates(text) == "ok\ud83d\ude00"
    # 无代理的普通文本原样返回
    assert sanitize_surrogates("plain ascii") == "plain ascii"
