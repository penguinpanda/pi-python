"""pi_ai.utils.validation 单元测试。

用例移植自 TS `packages/ai/test/validation.test.ts`，并补充
对象 / 数组 / 嵌套 / required / additionalProperties 等覆盖。
"""

from __future__ import annotations

import pytest

from pi_ai._types import Tool, ToolCall
from pi_ai.utils.validation import (
    ValidationError,
    coerce_with_json_schema,
    validate_arguments,
    validate_tool_call,
)


def _make_tool(schema: dict, name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description="Echo tool",
        input_schema={
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
        },
    )


def _make_tool_call(value: object, name: str = "echo") -> ToolCall:
    return ToolCall(
        type="toolCall",
        id="tool-1",
        name=name,
        arguments={"value": value},
        raw_arguments="",
    )


# ============================================================================
# 类型转换（coerce）—— 移植 TS validation.test.ts
# ============================================================================


class TestPrimitiveCoercion:
    """AJV 兼容原语转换规则。"""

    @pytest.mark.parametrize(
        ("schema", "input_value", "expected"),
        [
            ({"type": "number"}, "42", 42),
            ({"type": "number"}, True, 1),
            ({"type": "number"}, None, 0),
            ({"type": "integer"}, "42", 42),
            ({"type": "integer"}, True, 1),
            ({"type": "integer"}, None, 0),
            ({"type": "boolean"}, "true", True),
            ({"type": "boolean"}, "false", False),
            ({"type": "boolean"}, 1, True),
            ({"type": "boolean"}, 0, False),
            ({"type": "boolean"}, None, False),
            ({"type": "string"}, None, ""),
            ({"type": "string"}, True, "true"),
            ({"type": "null"}, "", None),
            ({"type": "null"}, 0, None),
            ({"type": "null"}, False, None),
            # 联合类型
            ({"type": ["number", "string"]}, "1", "1"),
            ({"type": ["boolean", "number"]}, "1", 1),
        ],
    )
    def test_coerces(self, schema, input_value, expected):
        tool = _make_tool(schema)
        tool_call = _make_tool_call(input_value)
        assert validate_tool_call([tool], tool_call) == {"value": expected}

    @pytest.mark.parametrize(
        ("schema", "input_value"),
        [
            ({"type": "boolean"}, "1"),
            ({"type": "boolean"}, "0"),
            ({"type": "null"}, "null"),
            ({"type": "integer"}, "42.1"),
        ],
    )
    def test_rejects_invalid_coercions(self, schema, input_value):
        tool = _make_tool(schema)
        tool_call = _make_tool_call(input_value)
        with pytest.raises(ValidationError, match="Validation failed"):
            validate_tool_call([tool], tool_call)

    def test_accepts_null_for_nullable_array(self):
        tool = _make_tool({"type": ["array", "null"], "items": {"type": "string"}})
        tool_call = _make_tool_call(None)
        assert validate_tool_call([tool], tool_call) == {"value": None}


# ============================================================================
# 对象 / 数组 / 嵌套
# ============================================================================


class TestObjectAndArray:
    def test_nested_object_coercion(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
            },
            "required": ["count"],
        }
        args = {"count": "3", "flag": "true"}
        assert coerce_with_json_schema(args, schema) == {"count": 3, "flag": True}

    def test_array_items_coercion(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        assert coerce_with_json_schema(["1", "2", "3"], schema) == [1, 2, 3]

    def test_tuple_items_coercion(self):
        schema = {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
        }
        # 位置 0 按 integer 转换 "1"→1；位置 1 按 string 转换 2→"2"
        assert coerce_with_json_schema(["1", 2], schema) == [1, "2"]

    def test_tuple_items_shorter_than_schema(self):
        """元组 schema 比值长时，越界位置跳过（不报错）。"""
        schema = {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
        }
        assert coerce_with_json_schema(["1"], schema) == [1]

    def test_tuple_items_validation(self):
        """校验阶段按元组 items 逐位置校验。"""
        schema = {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
        }
        assert validate_arguments("t", schema, [1, "x"]) == [1, "x"]
        # "abc" 无法转成 integer → 位置 0 校验失败
        with pytest.raises(ValidationError, match="Expected type integer"):
            validate_arguments("t", schema, ["abc", "x"])

    def test_validate_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
            },
            "required": ["path"],
        }
        assert validate_arguments("read", schema, {"path": "/tmp/a", "offset": "10"}) == {
            "path": "/tmp/a",
            "offset": 10,
        }

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        with pytest.raises(ValidationError, match="path: Required"):
            validate_arguments("read", schema, {})

    def test_nested_path_in_error_message(self):
        """嵌套对象校验失败时路径为 a.b.c 形式。"""
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "integer"}},
                    "required": ["inner"],
                }
            },
        }
        with pytest.raises(ValidationError, match="outer.inner: Required"):
            validate_arguments("read", schema, {"outer": {}})

    def test_wrong_type_error_message_contains_path(self):
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
        with pytest.raises(ValidationError, match=r"limit: Expected type"):
            validate_arguments("read", schema, {"limit": "abc"})

    def test_additional_properties_allowed_by_default(self):
        """未知属性默认放行（标准 JSON Schema；与既有 agent 测试兼容）。"""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert validate_arguments("t", schema, {"a": "x", "extra": 1}) == {
            "a": "x",
            "extra": 1,
        }

    def test_additional_properties_false_rejected(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValidationError, match="extra: Unexpected property"):
            validate_arguments("t", schema, {"a": "x", "extra": 1})

    def test_enum(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        assert validate_arguments("t", schema, "a") == "a"
        with pytest.raises(ValidationError, match="Expected one of"):
            validate_arguments("t", schema, "c")


# ============================================================================
# validate_tool_call / 错误格式
# ============================================================================


class TestValidateToolCall:
    def test_tool_not_found(self):
        tool = _make_tool({"type": "string"})
        with pytest.raises(ValidationError, match='Tool "nope" not found'):
            validate_tool_call([tool], _make_tool_call("x", name="nope"))

    def test_success_returns_coerced_arguments(self):
        tool = _make_tool({"type": "integer"})
        result = validate_tool_call([tool], _make_tool_call("42"))
        assert result == {"value": 42}

    def test_error_message_contains_received_arguments(self):
        tool = _make_tool({"type": "boolean"})
        with pytest.raises(ValidationError) as exc_info:
            validate_tool_call([tool], _make_tool_call("1"))
        message = str(exc_info.value)
        assert message.startswith('Validation failed for tool "echo":')
        assert "Received arguments:" in message
        assert '"value": "1"' in message

    def test_input_is_not_mutated(self):
        tool = _make_tool({"type": "integer"})
        original = {"value": "42"}
        validate_tool_call([tool], _make_tool_call(original["value"]))
        assert original == {"value": "42"}

    def test_none_arguments_treated_as_empty(self):
        tool = Tool(
            name="noop",
            description="No-op",
            input_schema={"type": "object", "properties": {}},
        )
        tool_call = ToolCall(
            type="toolCall",
            id="t1",
            name="noop",
            arguments=None,
            raw_arguments="",
        )
        assert validate_tool_call([tool], tool_call) == {}


# ============================================================================
# 高级 schema 结构（allOf / anyOf / oneOf / additionalProperties / 约束）
# ============================================================================


class TestAdvancedSchemas:
    def test_all_of_coercion(self):
        schema = {"allOf": [{"type": "number"}]}
        assert coerce_with_json_schema("42", schema) == 42

    def test_any_of_first_valid_wins(self):
        schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        # "42" 可转 integer 且通过校验 → 用第一个分支
        assert coerce_with_json_schema("42", schema) == 42

    def test_any_of_no_valid_returns_original(self):
        schema = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
        # "1" 无法转成 boolean（非 "true"/"false"），null 也不匹配 → 原样
        assert coerce_with_json_schema("1", schema) == "1"

    def test_one_of_coercion(self):
        schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        assert coerce_with_json_schema("7", schema) == 7

    def test_additional_properties_as_schema_coercion(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "boolean"},
        }
        # 已声明字段按 properties 转；未声明字段按 additionalProperties 转
        assert coerce_with_json_schema(
            {"a": "1", "flag": "true"}, schema
        ) == {"a": 1, "flag": True}

    def test_additional_properties_as_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "boolean"},
        }
        with pytest.raises(ValidationError, match="flag: Expected type"):
            validate_arguments("t", schema, {"a": 1, "flag": "not-bool-string"})

    def test_min_items(self):
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
        assert validate_arguments("t", schema, [1, 2]) == [1, 2]
        with pytest.raises(ValidationError, match="Expected at least 2 items"):
            validate_arguments("t", schema, [1])

    def test_string_length_constraints(self):
        assert validate_arguments("t", {"type": "string", "minLength": 2}, "abc") == "abc"
        with pytest.raises(ValidationError, match="Expected at least 2 characters"):
            validate_arguments("t", {"type": "string", "minLength": 2}, "a")
        with pytest.raises(ValidationError, match="Expected at most 2 characters"):
            validate_arguments("t", {"type": "string", "maxLength": 2}, "abc")

    def test_number_range_constraints(self):
        assert validate_arguments("t", {"type": "number", "minimum": 5}, 10) == 10
        with pytest.raises(ValidationError, match="Expected >= 5"):
            validate_arguments("t", {"type": "number", "minimum": 5}, 3)
        with pytest.raises(ValidationError, match="Expected <= 5"):
            validate_arguments("t", {"type": "number", "maximum": 5}, 10)

    def test_unknown_type_fails_validation(self):
        # _matches_json_type 对未知类型返回 False
        with pytest.raises(ValidationError, match="Expected type unknown"):
            validate_arguments("t", {"type": "unknown"}, "x")

    def test_non_numeric_strings_stay_unchanged(self):
        # number / integer 遇到不可解析字符串 → 保持原样 → 校验失败
        with pytest.raises(ValidationError, match="Expected type number"):
            validate_arguments("t", {"type": "number"}, "abc")
        with pytest.raises(ValidationError, match="Expected type integer"):
            validate_arguments("t", {"type": "integer"}, "abc")


# ============================================================================
# dict 形态 Tool / 直接调用
# ============================================================================


class TestDictTools:
    def test_validate_tool_call_with_dict_tools(self):
        """Tool 可以是带 name/input_schema 的 dict。"""
        tools = [
            {"name": "echo", "input_schema": {"type": "integer"}},
        ]
        tool_call = {
            "type": "toolCall",
            "id": "t1",
            "name": "echo",
            "arguments": "42",
            "raw_arguments": "",
        }
        assert validate_tool_call(tools, tool_call) == 42

    def test_validate_arguments_none_failure_message(self):
        """arguments=None 时错误消息里的 Received arguments 为空对象。"""
        with pytest.raises(ValidationError) as exc_info:
            validate_arguments("t", {"type": "string", "minLength": 5}, None)
        message = str(exc_info.value)
        assert "Received arguments:" in message
        assert "{}" in message


class TestExtendedKeywords:
    """新增 JSON Schema 关键字校验。"""

    def test_const(self):
        schema = {"type": "object", "properties": {"x": {"type": "string", "const": "fixed"}}}
        with pytest.raises(ValidationError, match="const"):
            validate_arguments("t", schema, {"x": "other"})
        assert validate_arguments("t", schema, {"x": "fixed"}) == {"x": "fixed"}

    def test_pattern(self):
        schema = {"type": "object", "properties": {"code": {"type": "string", "pattern": "^[a-z]+$"}}}
        with pytest.raises(ValidationError, match="pattern"):
            validate_arguments("t", schema, {"code": "ABC123"})
        assert validate_arguments("t", schema, {"code": "abc"}) == {"code": "abc"}

    def test_max_items(self):
        schema = {"type": "object", "properties": {"list": {"type": "array", "maxItems": 2}}}
        with pytest.raises(ValidationError, match="at most 2"):
            validate_arguments("t", schema, {"list": [1, 2, 3]})
        assert validate_arguments("t", schema, {"list": [1, 2]}) == {"list": [1, 2]}

    def test_min_max_properties(self):
        schema = {"type": "object", "minProperties": 1, "maxProperties": 2}
        with pytest.raises(ValidationError, match="at least 1"):
            validate_arguments("t", schema, {})
        with pytest.raises(ValidationError, match="at most 2"):
            validate_arguments("t", schema, {"a": 1, "b": 2, "c": 3})

    def test_multiple_of(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 5}}}
        with pytest.raises(ValidationError, match="multiple of 5"):
            validate_arguments("t", schema, {"n": 7})
        assert validate_arguments("t", schema, {"n": 10}) == {"n": 10}

    def test_unique_items(self):
        schema = {"type": "object", "properties": {"ids": {"type": "array", "uniqueItems": True}}}
        with pytest.raises(ValidationError, match="unique"):
            validate_arguments("t", schema, {"ids": ["a", "a"]})
        assert validate_arguments("t", schema, {"ids": ["a", "b"]}) == {"ids": ["a", "b"]}

    def test_exclusive_bounds(self):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 10}},
        }
        with pytest.raises(ValidationError, match="> 0"):
            validate_arguments("t", schema, {"n": 0})
        with pytest.raises(ValidationError, match="< 10"):
            validate_arguments("t", schema, {"n": 10})
        assert validate_arguments("t", schema, {"n": 5}) == {"n": 5}

    def test_not(self):
        schema = {"type": "object", "properties": {"name": {"not": {"enum": ["reserved"]}}}}
        with pytest.raises(ValidationError, match="not match"):
            validate_arguments("t", schema, {"name": "reserved"})
        assert validate_arguments("t", schema, {"name": "ok"}) == {"name": "ok"}

    def test_nested_paths_with_new_keywords(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[a-z]+$"},
                    "uniqueItems": True,
                }
            },
        }
        with pytest.raises(ValidationError, match=r"tags\.1"):
            validate_arguments("t", schema, {"tags": ["ok", "BAD"]})
        assert validate_arguments("t", schema, {"tags": ["ok", "fine"]}) == {"tags": ["ok", "fine"]}
