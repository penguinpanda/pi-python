"""配置值解析（对齐 TS core/resolve-config-value.ts）。

用于 models.json / auth.json 中的 API Key、header 等值：

- `!command`       执行命令并取 stdout（进程内缓存；无 shell 解释，不支持管道/重定向）；
- `$ENV_VAR` / `${ENV_VAR}`  引用环境变量（可出现在任意位置）；
- `$$` / `$!`      转义字面量 `$` / `!`（非命令值）；
- 其它             原样作为字面量。

任何引用缺失时解析结果为 None（调用方决定是否报错）。
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import TypedDict

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VAR_NAME_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")

# shell 命令结果缓存（进程生命周期内）。
_COMMAND_RESULT_CACHE: dict[str, str | None] = {}


class TemplatePart(TypedDict):
    type: str  # "literal" | "env"
    name: str
    value: str


def _append_literal(parts: list[TemplatePart], value: str) -> None:
    if not value:
        return
    previous = parts[-1] if parts else None
    if previous is not None and previous["type"] == "literal":
        previous["value"] += value
        return
    parts.append(TemplatePart(type="literal", name="", value=value))


def parse_config_value_template(config: str) -> list[TemplatePart]:
    """把配置值解析为 字面量/环境变量 序列（对齐 TS parseConfigValueTemplate）。"""
    parts: list[TemplatePart] = []
    index = 0
    while index < len(config):
        dollar_index = config.find("$", index)
        if dollar_index < 0:
            _append_literal(parts, config[index:])
            break

        _append_literal(parts, config[index:dollar_index])
        next_char = config[dollar_index + 1] if dollar_index + 1 < len(config) else ""

        if next_char in ("$", "!"):
            _append_literal(parts, next_char)
            index = dollar_index + 2
            continue

        if next_char == "{":
            end_index = config.find("}", dollar_index + 2)
            if end_index < 0:
                _append_literal(parts, "$")
                index = dollar_index + 1
                continue
            name = config[dollar_index + 2 : end_index]
            if _ENV_VAR_NAME_RE.match(name):
                parts.append(TemplatePart(type="env", name=name, value=""))
            else:
                _append_literal(parts, config[dollar_index : end_index + 1])
            index = end_index + 1
            continue

        match = _ENV_VAR_NAME_PREFIX_RE.match(config[dollar_index + 1 :])
        if match:
            parts.append(TemplatePart(type="env", name=match.group(0), value=""))
            index = dollar_index + 1 + len(match.group(0))
            continue

        _append_literal(parts, "$")
        index = dollar_index + 1

    return parts


def _parse_config_value_reference(config: str) -> tuple[bool, list[TemplatePart]]:
    """返回 (is_command, parts)。"""
    if config.startswith("!"):
        return True, []
    return False, parse_config_value_template(config)


def _resolve_env_value(name: str, env: dict[str, str] | None = None) -> str | None:
    value = env.get(name) if env else None
    if value is not None:
        return value
    return os.environ.get(name) or None


def _get_template_env_var_names(parts: list[TemplatePart]) -> list[str]:
    names: list[str] = []
    for part in parts:
        if part["type"] != "env" or part["name"] in names:
            continue
        names.append(part["name"])
    return names


def _resolve_template(parts: list[TemplatePart], env: dict[str, str] | None = None) -> str | None:
    resolved = ""
    for part in parts:
        if part["type"] == "literal":
            resolved += part["value"]
            continue
        env_value = _resolve_env_value(part["name"], env)
        if env_value is None:
            return None
        resolved += env_value
    return resolved


def get_config_value_env_var_name(config: str) -> str | None:
    """仅当值恰好是一个纯环境变量引用时返回其名称。"""
    is_command, parts = _parse_config_value_reference(config)
    if is_command:
        return None
    if len(parts) == 1 and parts[0]["type"] == "env":
        return parts[0]["name"]
    return None


def get_config_value_env_var_names(config: str) -> list[str]:
    """返回配置值引用的全部环境变量名（命令值返回空）。"""
    is_command, parts = _parse_config_value_reference(config)
    if is_command:
        return []
    return _get_template_env_var_names(parts)


def get_missing_config_value_env_var_names(
    config: str, env: dict[str, str] | None = None
) -> list[str]:
    """返回当前未设置的环境变量引用。"""
    return [
        name
        for name in get_config_value_env_var_names(config)
        if _resolve_env_value(name, env) is None
    ]


def is_command_config_value(config: str) -> bool:
    return config.startswith("!")


def is_config_value_configured(config: str, env: dict[str, str] | None = None) -> bool:
    return not get_missing_config_value_env_var_names(config, env)


def _execute_command_uncached(command_config: str) -> str | None:
    """执行 `!command` 并返回 stdout（去首尾空白）。

    参数经 shlex 拆分后直接执行，不经过 shell，
    因此 `|`、`&&`、`;` 等 shell 元字符不会被解释，
    避免配置值被当作任意 shell 脚本执行（命令注入边界）。
    配置来源仍应视为可信输入，命令受执行环境权限约束。
    """
    command = command_config[1:]
    parts = shlex.split(command, posix=(os.name != "nt"))
    if not parts:
        return None
    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _execute_command(command_config: str) -> str | None:
    if command_config in _COMMAND_RESULT_CACHE:
        return _COMMAND_RESULT_CACHE[command_config]
    result = _execute_command_uncached(command_config)
    _COMMAND_RESULT_CACHE[command_config] = result
    return result


def resolve_config_value(config: str, env: dict[str, str] | None = None) -> str | None:
    """解析配置值；无法解析（命令失败 / 环境变量缺失）时返回 None。"""
    is_command, parts = _parse_config_value_reference(config)
    if is_command:
        return _execute_command(config)
    return _resolve_template(parts, env)


def resolve_config_value_uncached(config: str, env: dict[str, str] | None = None) -> str | None:
    """同 resolve_config_value，但命令结果不缓存（供或错解析使用）。"""
    is_command, parts = _parse_config_value_reference(config)
    if is_command:
        return _execute_command_uncached(config)
    return _resolve_template(parts, env)


def resolve_config_value_or_throw(
    config: str, description: str, env: dict[str, str] | None = None
) -> str:
    """解析配置值，失败时抛出带描述的错误。"""
    resolved = resolve_config_value_uncached(config, env)
    if resolved is not None:
        return resolved

    is_command, _parts = _parse_config_value_reference(config)
    if is_command:
        raise ValueError(f"Failed to resolve {description} from shell command: {config[1:]}")

    missing = get_missing_config_value_env_var_names(config, env)
    if len(missing) == 1:
        raise ValueError(f"Failed to resolve {description} from environment variable: {missing[0]}")
    if len(missing) > 1:
        raise ValueError(
            f"Failed to resolve {description} from environment variables: {', '.join(missing)}"
        )
    raise ValueError(f"Failed to resolve {description}")


def resolve_headers(
    headers: dict[str, str] | None, env: dict[str, str] | None = None
) -> dict[str, str] | None:
    """解析全部 header 值（失败项跳过）。"""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value, env)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved or None


def resolve_headers_or_throw(
    headers: dict[str, str] | None,
    description: str,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """解析全部 header 值，任何一项失败即抛错。"""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = resolve_config_value_or_throw(value, f'{description} header "{key}"', env)
    return resolved or None


def clear_config_value_cache() -> None:
    """清空命令结果缓存（测试用）。"""
    _COMMAND_RESULT_CACHE.clear()


__all__ = [
    "TemplatePart",
    "parse_config_value_template",
    "get_config_value_env_var_name",
    "get_config_value_env_var_names",
    "get_missing_config_value_env_var_names",
    "is_command_config_value",
    "is_config_value_configured",
    "resolve_config_value",
    "resolve_config_value_uncached",
    "resolve_config_value_or_throw",
    "resolve_headers",
    "resolve_headers_or_throw",
    "clear_config_value_cache",
]
