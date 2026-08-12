"""Unicode 代理字符清洗（对齐 TS utils/sanitize-unicode.ts）。

未配对的高代理（0xD800-0xDBFF 后无低代理）或低代理（0xDC00-0xDFFF 前无
高代理）会导致部分 API provider 的 JSON 序列化失败；正确配对的代理
（emoji 等 BMP 外字符）不受影响。
"""

from __future__ import annotations

_HIGH_SURROGATE_LO = "\ud800"
_HIGH_SURROGATE_HI = "\udbff"
_LOW_SURROGATE_LO = "\udc00"
_LOW_SURROGATE_HI = "\udfff"


def sanitize_surrogates(text: str) -> str:
    """删除未配对的 Unicode 代理字符（对齐 TS sanitizeSurrogates）。"""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if _HIGH_SURROGATE_LO <= char <= _HIGH_SURROGATE_HI:
            if index + 1 < length and _LOW_SURROGATE_LO <= text[index + 1] <= _LOW_SURROGATE_HI:
                # 配对代理：原样保留
                out.append(char)
                out.append(text[index + 1])
                index += 2
            else:
                index += 1  # 孤立高代理：删除
        elif _LOW_SURROGATE_LO <= char <= _LOW_SURROGATE_HI:
            index += 1  # 孤立低代理：删除
        else:
            out.append(char)
            index += 1
    return "".join(out)


__all__ = ["sanitize_surrogates"]
