"""手写部分 JSON 解析器（对齐 TS 的 partial-json 行为）。

输入可能是流式中截断/不完整的 JSON 文本，输出"尽可能完整"的解析结果：

- 完整文本直接 `json.loads`；
- 截断文本：找到最后一个完整值边界，丢弃尾部不完整 token，
  自动补齐未闭合的字符串引号与容器括号；
- 完全无法解析时返回 {}。
"""

import json

from typing import Any

_WHITESPACE = " \t\n\r"
_LITERALS = ("true", "false", "null")


def _number_prefix_length(token: str) -> int:
    """返回 token 中最长合法 JSON 数字前缀的长度；无合法前缀返回 0。"""
    for end in range(len(token), 0, -1):
        try:
            json.loads(token[:end])
        except (ValueError, TypeError):
            continue
        return end
    return 0


def _close_containers(text: str, stack: list[str]) -> str:
    """为未闭合的容器补齐闭合括号。"""
    result = text.rstrip()
    for container in reversed(stack):
        result += "}" if container == "{" else "]"
    return result


def _scan(
    text: str,
) -> tuple[list[tuple[int, list[str]]], list[str], bool, int]:
    """扫描 partial JSON，返回 (candidates, stack, in_string, n)。

    candidates：[(cut, stack_snapshot), ...]，每个完整值边界的截断位置
    （含位置时的容器栈快照，用于只补齐该位置之前打开的容器）。
    """
    n = len(text)
    in_string = False
    escape = False
    stack: list[str] = []
    expect_value = True
    candidates: list[tuple[int, list[str]]] = []
    i = 0

    while i < n:
        ch = text[i]

        # 字符串内部：只关心转义与闭合引号。
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                # 字符串作为"值"闭合时，才是一个完整值边界
                # （作为对象 key 闭合时不是，稍后由 json.loads 失败剔除）。
                if expect_value:
                    candidates.append((i + 1, list(stack)))
                    expect_value = False
            i += 1
            continue

        if ch in _WHITESPACE:
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in "[{":
            stack.append(ch)
            expect_value = True
            i += 1
            continue
        if ch in "]}":
            if stack:
                stack.pop()
                candidates.append((i + 1, list(stack)))
                expect_value = False
            i += 1
            continue
        if ch == ",":
            expect_value = True
            candidates.append((i + 1, list(stack)))
            i += 1
            continue
        if ch == ":":
            expect_value = True
            i += 1
            continue

        # 数字（可能是不完整数字，如 "1." / "-" / "12e"）。
        if ch == "-" or ch.isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] in ".-+eE"):
                j += 1
            k = _number_prefix_length(text[i:j])
            if k > 0 and expect_value:
                candidates.append((i + k, list(stack)))
                expect_value = False
            i = j
            continue

        # true / false / null 字面量（可能不完整，如 "tru"）。
        matched = False
        for literal in _LITERALS:
            if text.startswith(literal, i):
                if expect_value:
                    candidates.append((i + len(literal), list(stack)))
                    expect_value = False
                i += len(literal)
                matched = True
                break
        if matched:
            continue

        # 未知字符 / 不完整字面量：停止扫描，之前的边界即最大有效前缀。
        break

    return candidates, stack, in_string, n


def partial_json(text: str) -> Any:
    """解析（可能不完整的）JSON 文本，返回尽量完整的结果；失败返回 {}。"""
    if text is None:
        return {}
    if not text.strip():
        return {}

    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    candidates, stack, in_string, n = _scan(text)
    prefixes: list[str] = []

    # 1) 字符串未闭合：补引号后再补容器。
    if in_string:
        prefixes.append(_close_containers(text + '"', stack))

    # 2) 从最新的完整值边界开始向前回退尝试（越新越优先）。
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for cut, stack_snapshot in reversed(candidates):
        key = (cut, tuple(stack_snapshot))
        if key in seen or cut <= 0 or cut > n:
            continue
        seen.add(key)
        candidate = text[:cut].rstrip()
        while candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        prefixes.append(_close_containers(candidate, stack_snapshot))

    # 3) 兜底：原文直接补容器。
    prefixes.append(_close_containers(text, stack))

    for candidate in prefixes:
        if not candidate.strip():
            continue
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return {}


__all__ = ["partial_json"]
