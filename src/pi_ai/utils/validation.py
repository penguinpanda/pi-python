"""pi_ai.utils.validation — 工具调用参数校验与类型转换。

对应 TypeScript `packages/ai/src/utils/validation.ts` 的移植：

    validateToolCall      → validate_tool_call
    validateToolArguments → validate_tool_arguments
    coerceWithJsonSchema  → coerce_with_json_schema

TS 端基于 TypeBox（编译期 schema）+ 手写 coerce；Python 端用轻量
手写校验器（无新依赖），覆盖 coding-agent 工具实际使用的 JSON Schema
子集：

    type / required / properties / items / enum /
    additionalProperties / minLength / maxLength / minimum / maximum /
    const / pattern / maxItems / minProperties / maxProperties /
    multipleOf / uniqueItems / exclusiveMinimum / exclusiveMaximum / not

行为对齐 TS：

    ① 类型转换（AJV 兼容原语规则）：

        "42"    → 42
        true    → 1
        null    → 0
        "true"  → True
        "1"     → 字符串保持原样（boolean 拒绝非 "true"/"false"）
        "42.1"  → integer 拒绝（保持原样 → 校验失败）
        null    → ""（string）、null（null 类型）

    ② 校验失败抛 ValidationError，携带字段路径与收到的参数。

    ③ 未知属性默认放行（标准 JSON Schema 语义）；schema 显式声明
       `additionalProperties: false` 时才拒绝。这与既有 agent 测试
       兼容（测试工具 schema 为 `properties: {}`，参数可含任意键）。
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

# =========================================================
# 异常
# =========================================================


class ValidationError(ValueError):
    """工具参数校验失败。

    消息格式（对齐 TS）：

        Validation failed for tool "<name>":
          - <path>: <message>
          ...
        <空行>
        Received arguments:
        {json}
    """


# =========================================================
# 类型判断
# =========================================================


def _schema_types(schema: dict[str, Any]) -> list[str]:
    """提取 schema 的 type 列表（字符串或字符串数组）。"""
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def _matches_json_type(value: Any, type_name: str) -> bool:
    """判断 value 是否匹配 JSON Schema 类型（bool 不算 number/integer）。"""
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "null":
        return value is None
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


# =========================================================
# 类型转换（coerce）
# =========================================================


def _coerce_primitive_by_type(value: Any, type_name: str) -> Any:
    """按 JSON Schema 类型转换单个原语（AJV 兼容规则，对齐 TS）。

    无法转换时原样返回（由后续校验判定失败）。
    """
    if type_name == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            try:
                parsed = float(value)
            except ValueError:
                return value
            if math.isfinite(parsed):
                return int(parsed) if parsed.is_integer() else parsed
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if type_name == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            try:
                parsed = float(value)
            except ValueError:
                return value
            # Number.isInteger 语义：42.1 不转换（保持原样 → 校验失败）
            if math.isfinite(parsed) and parsed.is_integer():
                return int(parsed)
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if type_name == "boolean":
        if value is None:
            return False
        if isinstance(value, str):
            if value == "true":
                return True
            if value == "false":
                return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value

    if type_name == "string":
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return value

    if type_name == "null":
        if value == "" or value == 0 or value is False:
            return None
        return value

    return value


def _coerce_with_union_schema(value: Any, schemas: list[dict[str, Any]]) -> Any:
    """anyOf / oneOf：逐个尝试转换，第一个校验通过的返回。"""
    for schema in schemas:
        candidate = copy.deepcopy(value)
        coerced = coerce_with_json_schema(candidate, schema)
        if _check_schema(coerced, schema) == []:
            return coerced
    return value


def _apply_schema_object_coercion(value: dict[str, Any], schema: dict[str, Any]) -> None:
    """对象：按 properties 递归转换每个已声明字段；additionalProperties 为
    schema 时转换未声明字段（原地修改，对齐 TS 的 applySchemaObjectCoercion）。"""
    properties = schema.get("properties")
    defined_keys: set[str] = set(properties.keys()) if isinstance(properties, dict) else set()

    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key in value:
                value[key] = coerce_with_json_schema(value[key], property_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, property_value in value.items():
            if key in defined_keys:
                continue
            value[key] = coerce_with_json_schema(property_value, additional)


def _apply_schema_array_coercion(value: list[Any], schema: dict[str, Any]) -> None:
    """数组：按 items（单个 schema 或元组 schema）递归转换每个元素。"""
    items = schema.get("items")

    if isinstance(items, list):
        for index, item_schema in enumerate(items):
            if index >= len(value) or not isinstance(item_schema, dict):
                continue
            value[index] = coerce_with_json_schema(value[index], item_schema)
        return

    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = coerce_with_json_schema(value[index], items)


def coerce_with_json_schema(value: Any, schema: dict[str, Any]) -> Any:
    """按 JSON Schema 递归转换值（对齐 TS coerceWithJsonSchema）。

    顺序：
        ① allOf / anyOf / oneOf
        ② 原语类型转换（含联合类型）
        ③ 对象字段 / 数组元素递归转换
    """
    next_value = value

    if isinstance(schema.get("allOf"), list):
        for nested in schema["allOf"]:
            next_value = coerce_with_json_schema(next_value, nested)

    if isinstance(schema.get("anyOf"), list):
        next_value = _coerce_with_union_schema(next_value, schema["anyOf"])

    if isinstance(schema.get("oneOf"), list):
        next_value = _coerce_with_union_schema(next_value, schema["oneOf"])

    schema_types = _schema_types(schema)
    # 联合类型：只要匹配任一成员则不再转换；否则尝试第一个能产生变化的转换。
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, t) for t in schema_types
    )
    if schema_types and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive_by_type(next_value, schema_type)
            # 注意：不能用 `!=` 判断是否变化 —— Python 中 1 == True / 0 == False，
            # 会漏掉 bool↔number 的转换。未转换时函数原样返回同一对象，
            # 因此用 `is not` 精确检测"产生了新值"。
            if candidate is not next_value:
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


# =========================================================
# 校验
# =========================================================


def _join_path(path: str, key: str) -> str:
    """拼接校验路径（root 开头时不带前缀）。"""
    if path == "root":
        return key
    return f"{path}.{key}"


def _check_schema(
    value: Any,
    schema: dict[str, Any],
    path: str = "root",
) -> list[tuple[str, str]]:
    """递归校验 value 是否满足 schema。返回 [(path, message), ...]。"""
    errors: list[tuple[str, str]] = []

    schema_types = _schema_types(schema)
    if schema_types:
        if not any(_matches_json_type(value, t) for t in schema_types):
            errors.append((path, f"Expected type {schema.get('type')}"))
            # 类型不匹配时不再深入（properties/items 无意义）
            return errors

    # const
    if "const" in schema and value != schema["const"]:
        errors.append((path, f"Expected const {schema['const']!r}"))

    # enum
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append((path, f"Expected one of {enum}"))

    # not：值满足嵌套 schema 时失败。
    not_schema = schema.get("not")
    if isinstance(not_schema, dict) and _check_schema(value, not_schema, path) == []:
        errors.append((path, "Expected value to not match schema"))

    # object
    if isinstance(value, dict):
        properties = schema.get("properties")

        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key in value:
                    errors.extend(_check_schema(value[key], property_schema, _join_path(path, key)))

        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append((_join_path(path, key), "Required"))

        defined_keys = set(properties.keys()) if isinstance(properties, dict) else set()
        additional = schema.get("additionalProperties")
        if additional is False:
            for key in value:
                if key not in defined_keys:
                    errors.append((_join_path(path, key), "Unexpected property"))
        elif isinstance(additional, dict):
            for key, property_value in value.items():
                if key in defined_keys:
                    continue
                errors.extend(_check_schema(property_value, additional, _join_path(path, key)))

    # array
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(_check_schema(item, items, _join_path(path, str(index))))
        elif isinstance(items, list):
            for index, item_schema in enumerate(items):
                if index < len(value) and isinstance(item_schema, dict):
                    errors.extend(
                        _check_schema(value[index], item_schema, _join_path(path, str(index)))
                    )

        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append((path, f"Expected at least {min_items} items"))
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append((path, f"Expected at most {max_items} items"))
        if schema.get("uniqueItems") is True:
            seen: list[Any] = []
            for item in value:
                if item in seen:
                    errors.append((path, "Expected unique items"))
                    break
                seen.append(item)

    # object 尺寸约束
    if isinstance(value, dict):
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            errors.append((path, f"Expected at least {min_properties} properties"))
        max_properties = schema.get("maxProperties")
        if isinstance(max_properties, int) and len(value) > max_properties:
            errors.append((path, f"Expected at most {max_properties} properties"))

    # string 约束
    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append((path, f"Expected at least {min_length} characters"))
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append((path, f"Expected at most {max_length} characters"))
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            if re.search(pattern, value) is None:
                errors.append((path, f"Expected pattern {pattern!r}"))

    # number 约束
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append((path, f"Expected >= {minimum}"))
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append((path, f"Expected <= {maximum}"))
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            errors.append((path, f"Expected > {exclusive_minimum}"))
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            errors.append((path, f"Expected < {exclusive_maximum}"))
        multiple_of = schema.get("multipleOf")
        if isinstance(multiple_of, (int, float)) and multiple_of > 0:
            if value % multiple_of != 0:
                errors.append((path, f"Expected multiple of {multiple_of}"))

    return errors


# =========================================================
# 公共 API
# =========================================================


def _tool_name(tool: Any) -> str:
    """读取 Tool（dataclass 或 dict）的名称。"""
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _tool_schema(tool: Any) -> dict[str, Any]:
    """读取 Tool（dataclass 或 dict）的 input_schema。"""
    if isinstance(tool, dict):
        return dict(tool.get("input_schema") or {})
    return dict(getattr(tool, "input_schema", None) or {})


def _format_validation_error(
    tool_name: str,
    errors: list[tuple[str, str]],
    raw_arguments: Any,
) -> str:
    """构造与 TS 一致的校验错误消息。"""
    lines = "\n".join(f"  - {path}: {message}" for path, message in errors)
    received = json.dumps(
        raw_arguments if raw_arguments is not None else {},
        indent=2,
        ensure_ascii=False,
    )
    return f'Validation failed for tool "{tool_name}":\n{lines}\n\nReceived arguments:\n{received}'


def validate_arguments(
    tool_name: str,
    schema: dict[str, Any],
    arguments: Any,
) -> dict[str, Any]:
    """按 schema 校验工具调用参数（不依赖 Tool 对象，供 Agent 层使用）。

    流程（对齐 TS validateToolArguments）：
        ① 深拷贝参数（不修改原对象）
        ② 类型转换（coerce）
        ③ 校验；失败抛 ValidationError
        ④ 返回转换后的参数

    参数:
        tool_name: 工具名（用于错误消息）
        schema: JSON Schema（Tool.input_schema）
        arguments: LLM 返回的参数（通常为 dict；None 视为 {}）

    返回:
        转换后的参数字典。

    抛出:
        ValidationError: 校验失败（含字段路径与收到的参数）。
    """
    args = copy.deepcopy(arguments if arguments is not None else {})
    coerced = coerce_with_json_schema(args, schema)

    errors = _check_schema(coerced, schema)
    if errors:
        raise ValidationError(_format_validation_error(tool_name, errors, arguments))
    return coerced


def validate_tool_arguments(tool: Any, tool_call: dict[str, Any]) -> dict[str, Any]:
    """校验单个工具调用（对齐 TS validateToolArguments）。

    tool 可为 pi_ai 的 Tool（dataclass，含 input_schema 属性）
    或带 name/input_schema 的任意对象。
    tool_call 为 ToolCall（含 name / arguments 字段）。
    """
    return validate_arguments(
        _tool_name(tool),
        _tool_schema(tool),
        tool_call.get("arguments"),
    )


def validate_tool_call(tools: list[Any], tool_call: dict[str, Any]) -> dict[str, Any]:
    """按名称查找工具并校验其调用（对齐 TS validateToolCall）。

    抛出:
        ValidationError: 工具不存在，或参数校验失败。
    """
    name = tool_call.get("name")
    for tool in tools:
        if _tool_name(tool) == name:
            return validate_tool_arguments(tool, tool_call)
    raise ValidationError(f'Tool "{name}" not found')


__all__ = [
    "ValidationError",
    "coerce_with_json_schema",
    "validate_arguments",
    "validate_tool_arguments",
    "validate_tool_call",
]
