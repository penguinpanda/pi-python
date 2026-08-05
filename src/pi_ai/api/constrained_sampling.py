"""约束采样（对齐 TS api/constrained-sampling.ts）。

- json_schema 约束：决定请求是否开启 strict 工具模式；
- grammar 约束：从 Tool.constrainedSampling.variants 选择 Lark/正则定义，
  并支持把流式 grammar 输入拼装为 `{"prop":"..."}` JSON。
"""

import json

from dataclasses import dataclass
from typing import Any

from ..types import Tool


@dataclass(slots=True)
class GrammarConstrainedSampling:
    """已解析的 grammar 约束（lark 或 regex）。"""

    format: str  # "lark" | "regex"
    definition: str
    input_property: str


@dataclass(slots=True)
class GrammarToolInputJsonBuffer:
    """grammar tool 输入的流式累积缓冲。"""

    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(
    tool_name: str,
    arguments: dict[str, Any],
    input_property: str,
) -> str:
    """从工具调用参数中取出 grammar 输入（必须为字符串）。"""
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(
            f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.'
        )
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    """把流式 grammar 输入拼装为 `{"prop":"..."}` JSON 增量。

    要求 next_input 单调前缀扩展；closed 后不允许再变化。
    """
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed after it was closed'
        )
    if not next_input.startswith(buffer.input):
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed non-monotonically'
        )

    input_delta = next_input[len(buffer.input) :]
    if not close and len(input_delta) == 0:
        return None

    delta = ""
    if not buffer.started:
        delta += "{" + json.dumps(input_property, ensure_ascii=False) + ':"'
        buffer.started = True
    delta += json.dumps(input_delta, ensure_ascii=False)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def infer_grammar_input_property(tool: Tool) -> str:
    """从工具参数 schema 推断 grammar 输入属性。

    要求：object schema、恰好一个 required 属性、该属性类型为 string。
    """
    schema = tool.input_schema
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError(
            "grammar constrained sampling requires exactly one required string property"
        )
    input_property = required[0]
    properties = schema.get("properties")
    if not isinstance(properties, dict) or input_property not in properties:
        raise ValueError(
            f"grammar constrained sampling requires a properties entry for {input_property}"
        )
    prop = properties[input_property]
    if not isinstance(prop, dict) or prop.get("type") != "string":
        raise ValueError(
            f"grammar constrained sampling property {input_property} must have type string"
        )
    return input_property


def resolve_json_schema_strict_sampling(
    tool: Tool,
    supports_strict_mode: bool,
) -> bool | None:
    """json_schema 约束 → strict 开关；require 且不支持时抛错。"""
    config = tool.constrained_sampling
    if not config or not isinstance(config, dict) or config.get("type") != "json_schema":
        return None
    if supports_strict_mode:
        return True
    if config.get("strict") == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, '
            "but strict tools are unsupported."
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    """grammar 约束 → 选择 openai_lark / openai_regex 变体并推断输入属性。"""
    config = tool.constrained_sampling
    if not config or not isinstance(config, dict) or config.get("type") != "grammar":
        return None
    if not supports_openai_grammar_tools:
        return None

    variants = config.get("variants") or {}
    if not isinstance(variants, dict):
        variants = {}
    lark_definition = variants.get("openai_lark")
    regex_definition = variants.get("openai_regex")
    has_lark = isinstance(lark_definition, str) and lark_definition.strip() != ""
    has_regex = isinstance(regex_definition, str) and regex_definition.strip() != ""
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: '
            "no supported grammar variant was provided."
        )

    try:
        if has_lark:
            definition = lark_definition
        else:
            definition = regex_definition
        assert isinstance(definition, str)
        return GrammarConstrainedSampling(
            format="lark" if has_lark else "regex",
            definition=definition,
            input_property=infer_grammar_input_property(tool),
        )
    except ValueError as exc:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: {exc}'
        ) from exc


def create_grammar_tool_input_properties(
    tools: list[Tool] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    """返回 {tool_name: input_property} 映射（仅含可用 grammar 的工具）。"""
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar:
            properties[tool.name] = grammar.input_property
    return properties


__all__ = [
    "GrammarConstrainedSampling",
    "GrammarToolInputJsonBuffer",
    "get_grammar_tool_input",
    "append_grammar_tool_input_json_delta",
    "infer_grammar_input_property",
    "resolve_json_schema_strict_sampling",
    "resolve_grammar_constrained_sampling",
    "create_grammar_tool_input_properties",
]
