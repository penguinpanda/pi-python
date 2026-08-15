"""JSON 修复与流式解析（对齐 TS packages/ai/src/utils/json-parse.ts）。"""

import json

from typing import Any

from .partial_json import _finite_constant, _finite_float, partial_json

_VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _is_control_character(char: str) -> bool:
    return ord(char) <= 0x1F


def _escape_control_character(char: str) -> str:
    mapping = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if char in mapping:
        return mapping[char]
    return f"\\u{ord(char):04x}"


def repair_json(json_str: str) -> str:
    """修复字符串字面量中的原始控制字符与非法转义。

    与 TS 一致：
    - 字符串内的原始控制字符转义为 \\n / \\uXXXX 等；
    - 非法转义序列（\\x、\\ 后跟非合法转义字符）的反斜杠加倍。
    """
    repaired = ""
    in_string = False
    index = 0
    n = len(json_str)

    while index < n:
        char = json_str[index]

        if not in_string:
            repaired += char
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            repaired += char
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = json_str[index + 1] if index + 1 < n else None
            if next_char is None:
                repaired += "\\\\"
                index += 1
                continue
            if next_char == "u":
                unicode_digits = json_str[index + 2 : index + 6]
                if len(unicode_digits) == 4 and all(
                    c in "0123456789abcdefABCDEF" for c in unicode_digits
                ):
                    repaired += f"\\u{unicode_digits}"
                    index += 6
                    continue
            if next_char in _VALID_JSON_ESCAPES:
                repaired += f"\\{next_char}"
                index += 2
                continue
            repaired += "\\\\"
            index += 1
            continue

        repaired += _escape_control_character(char) if _is_control_character(char) else char
        index += 1

    return repaired


def parse_json_with_repair(json_str: str) -> Any:
    """JSON.parse → 失败则 repairJson 后重试；仍失败抛原异常。"""
    try:
        # parse_float/parse_constant 拒绝非有限数值（1e999 → inf、
        # Infinity/NaN 字面量），否则产出无法再序列化的非法 JSON。
        return json.loads(
            json_str,
            parse_float=_finite_float,
            parse_constant=_finite_constant,
        )
    except (ValueError, TypeError):
        repaired = repair_json(json_str)
        if repaired != json_str:
            return json.loads(
                repaired,
                parse_float=_finite_float,
                parse_constant=_finite_constant,
            )
        raise


def parse_streaming_json(partial: str | None) -> Any:
    """解析流式（可能不完整）JSON；永不抛异常，失败返回 {}。"""
    if partial is None or not partial.strip():
        return {}

    try:
        return parse_json_with_repair(partial)
    except Exception:
        pass

    try:
        return partial_json(partial)
    except Exception:
        pass

    try:
        return partial_json(repair_json(partial))
    except Exception:
        return {}


__all__ = ["repair_json", "parse_json_with_repair", "parse_streaming_json"]
