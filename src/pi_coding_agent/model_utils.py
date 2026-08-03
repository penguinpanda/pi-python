"""模型工具函数（对齐 TS pi-ai models.ts 的公开辅助）。

- THINKING_LEVELS / DEFAULT_THINKING_LEVEL；
- get_supported_thinking_levels / clamp_thinking_level；
- models_are_equal。
"""

from __future__ import annotations

from pi_ai import Model
from pi_ai.types.common import ModelThinkingLevel, ThinkingLevel

# 会话层可用的全部思考级别（扩展集：xhigh/max 需模型显式支持）。
THINKING_LEVELS: list[ModelThinkingLevel] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
]

DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"

# 完整思考级别顺序（用于 clamp 时向上/向下查找）。
_EXTENDED_THINKING_LEVELS: list[ModelThinkingLevel] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """按 provider + id 判断两个模型是否相同。"""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    """返回模型支持的思考级别。"""
    if not model.reasoning:
        return ["off"]
    mapping = model.thinking_level_map or {}
    supported: list[ModelThinkingLevel] = []
    for level in _EXTENDED_THINKING_LEVELS:
        mapped = mapping.get(level)
        if mapped is None:
            if level in ("xhigh", "max"):
                # xhigh/max 必须显式映射（非 null）才可用。
                continue
            if level in mapping:
                # 显式 null → 不支持。
                continue
        supported.append(level)
    return supported


def clamp_thinking_level(
    model: Model, level: ModelThinkingLevel
) -> ModelThinkingLevel:
    """把思考级别收敛到模型支持的范围内。"""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    if level not in _EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"

    requested_index = _EXTENDED_THINKING_LEVELS.index(level)
    # 向上找更高级别。
    for candidate in _EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    # 向下找较低级别。
    for candidate in reversed(_EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


__all__ = [
    "THINKING_LEVELS",
    "DEFAULT_THINKING_LEVEL",
    "models_are_equal",
    "get_supported_thinking_levels",
    "clamp_thinking_level",
]
