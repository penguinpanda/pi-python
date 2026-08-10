"""Deferred Tools 拆分（对齐 TS `utils/deferred-tools.ts`）。

按上下文中的历史工具调用把工具分成"立即定义"与"延迟定义"两组：
延迟定义的工具是工具结果中出现、但此前从未在 assistant toolCall 里出现过的
工具（例如预注册工具），调用方可在结果回传前不重复声明其完整定义。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ..types import Context, Tool


ToolNameNormalizer = Callable[[str], str]


def _identity(name: str) -> str:
    return name


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: ToolNameNormalizer = _identity,
) -> tuple[list[Tool], dict[str, Tool]]:
    """返回 (immediate, deferred)；deferred 按规范化工具名映射。"""
    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[normalize_name(tool.name)] = tool
    if not enabled:
        return list(unique_tools.values()), {}

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        if message["role"] == "assistant":
            for block in cast(list, message.get("content") or []):
                if block.get("type") == "toolCall":
                    used_names.add(normalize_name(block.get("name", "")))
        elif message["role"] == "toolResult":
            for name in message.get("added_tool_names") or []:
                normalized = normalize_name(name)
                if normalized not in used_names:
                    deferred_names.add(normalized)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)
    return immediate, deferred


__all__ = ["split_deferred_tools"]
