"""约束采样测试（对齐 TS api/constrained-sampling.ts）。"""

import pytest

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    infer_grammar_input_property,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pi_ai.api._shared import to_openai_tools
from pi_ai.types import Tool


def _grammar_tool() -> Tool:
    return Tool(
        name="math",
        description="Evaluate an expression",
        input_schema={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
        constrained_sampling={
            "type": "grammar",
            "variants": {
                "openai_lark": "expr: /[0-9+\\-*\\/ ]+/",
                "openai_regex": "^[0-9+\\-*\\/ ]+$",
            },
        },
    )


def _json_schema_tool(strict: str) -> Tool:
    return Tool(
        name="struct",
        description="Structured output",
        input_schema={"type": "object", "properties": {}},
        constrained_sampling={"type": "json_schema", "strict": strict},
    )


# ---------- json_schema strict ----------


def test_json_schema_strict_supported():
    assert resolve_json_schema_strict_sampling(_json_schema_tool("prefer"), True) is True
    assert resolve_json_schema_strict_sampling(_json_schema_tool("require"), True) is True


def test_json_schema_strict_unsupported_prefer():
    assert resolve_json_schema_strict_sampling(_json_schema_tool("prefer"), False) is None


def test_json_schema_strict_require_unsupported_raises():
    with pytest.raises(ValueError, match="strict tools are unsupported"):
        resolve_json_schema_strict_sampling(_json_schema_tool("require"), False)


def test_json_schema_strict_ignored_when_disabled():
    tool = _grammar_tool()
    assert resolve_json_schema_strict_sampling(tool, True) is None


# ---------- grammar ----------


def test_grammar_prefers_lark():
    result = resolve_grammar_constrained_sampling(_grammar_tool(), True)
    assert result is not None
    assert result.format == "lark"
    assert result.input_property == "expr"
    assert "expr:" in result.definition


def test_grammar_falls_back_to_regex():
    tool = _grammar_tool()
    tool.constrained_sampling = {
        "type": "grammar",
        "variants": {"openai_regex": "^x$"},
    }
    result = resolve_grammar_constrained_sampling(tool, True)
    assert result is not None
    assert result.format == "regex"


def test_grammar_unsupported_returns_none():
    assert resolve_grammar_constrained_sampling(_grammar_tool(), False) is None


def test_grammar_no_variant_raises():
    tool = _grammar_tool()
    tool.constrained_sampling = {"type": "grammar", "variants": {}}
    with pytest.raises(ValueError, match="no supported grammar variant"):
        resolve_grammar_constrained_sampling(tool, True)


def test_grammar_invalid_schema_raises():
    tool = _grammar_tool()
    tool.input_schema = {"type": "object", "properties": {}, "required": []}
    with pytest.raises(ValueError, match="exactly one required string property"):
        resolve_grammar_constrained_sampling(tool, True)


def test_infer_grammar_input_property():
    assert infer_grammar_input_property(_grammar_tool()) == "expr"


# ---------- 流式 delta 拼装 ----------


def test_append_grammar_delta_sequence():
    buffer = GrammarToolInputJsonBuffer()
    assert append_grammar_tool_input_json_delta(buffer, "expr", "1", False) == '{"expr":"1'
    assert append_grammar_tool_input_json_delta(buffer, "expr", "1+2", False) == "+2"
    assert append_grammar_tool_input_json_delta(buffer, "expr", "1+2", True) == '"}'
    assert buffer.closed is True


def test_append_grammar_delta_noop_when_no_change():
    buffer = GrammarToolInputJsonBuffer()
    assert append_grammar_tool_input_json_delta(buffer, "expr", "1", False) == '{"expr":"1'
    assert append_grammar_tool_input_json_delta(buffer, "expr", "1", False) is None


def test_append_grammar_delta_non_monotonic_raises():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "expr", "12", False)
    with pytest.raises(ValueError, match="non-monotonically"):
        append_grammar_tool_input_json_delta(buffer, "expr", "21", False)


def test_append_grammar_delta_closed_change_raises():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "expr", "1", True)
    with pytest.raises(ValueError, match="after it was closed"):
        append_grammar_tool_input_json_delta(buffer, "expr", "12", False)


def test_get_grammar_tool_input():
    assert get_grammar_tool_input("math", {"expr": "1+1"}, "expr") == "1+1"
    with pytest.raises(ValueError, match="to be a string"):
        get_grammar_tool_input("math", {"expr": 42}, "expr")


def test_create_grammar_tool_input_properties():
    tools = [_grammar_tool()]
    assert create_grammar_tool_input_properties(tools, True) == {"math": "expr"}
    assert create_grammar_tool_input_properties(tools, False) == {}


# ---------- to_openai_tools strict 接入 ----------


def test_to_openai_tools_adds_strict_when_supported():
    tools = [_json_schema_tool("prefer")]
    result = to_openai_tools(tools, supports_strict_mode=True)
    assert result[0]["function"]["strict"] is True


def test_to_openai_tools_no_strict_when_unsupported():
    tools = [_json_schema_tool("prefer")]
    result = to_openai_tools(tools, supports_strict_mode=False)
    assert "strict" not in result[0]["function"]


def test_to_openai_tools_plain_tools_unchanged():
    tool = Tool(name="x", description="d", input_schema={"type": "object"})
    result = to_openai_tools([tool])
    assert result[0]["function"]["name"] == "x"
    assert "strict" not in result[0]["function"]
