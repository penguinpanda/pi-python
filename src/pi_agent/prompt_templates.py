"""提示模板系统（Phase 4.5）。

对齐 TS `harness/prompt-templates.ts`：

- 从 .md 文件（目录递归）加载模板，frontmatter 提供 description
- 参数替换：$1..$N、$@ / $ARGUMENTS、${@:N}、${@:N:L}
- 多来源加载（source 原样保留）
"""

from __future__ import annotations

import re
import os
from typing import Any

from .env import ExecutionEnv
from .skills import _basename, _parse_frontmatter


class PromptTemplateDiagnostic:
    def __init__(self, code: str, message: str, path: str, type: str = "warning") -> None:
        self.type = type
        self.code = code
        self.message = message
        self.path = path


def substitute_args(content: str, args: list[str]) -> str:
    """替换模板参数（逐字对齐 TS substituteArgs）。"""
    result = content

    def _positional(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    result = re.sub(r"\$(\d+)", _positional, result)

    def _slice_args(match: re.Match[str]) -> str:
        start = int(match.group(1)) - 1
        if start < 0:
            start = 0
        if match.group(2) is not None:
            return " ".join(args[start : start + int(match.group(2))])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", _slice_args, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(
    name: str,
    content: str,
    args: list[str] | None = None,
) -> str:
    return substitute_args(content, args or [])


async def _load_template_from_file(
    env: ExecutionEnv,
    file_path: str,
) -> tuple[dict[str, Any] | None, list[PromptTemplateDiagnostic]]:
    diagnostics: list[PromptTemplateDiagnostic] = []
    raw = await env.read_text_file(file_path)
    if not raw[0]:
        diagnostics.append(PromptTemplateDiagnostic("read_failed", raw[1].message, file_path))
        return None, diagnostics
    try:
        frontmatter, body = _parse_frontmatter(raw[1])
    except Exception as error:
        diagnostics.append(PromptTemplateDiagnostic("parse_failed", str(error), file_path))
        return None, diagnostics
    description = frontmatter.get("description")
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        base = _basename(file_path)
        name = base[: -len(".md")] if base.endswith(".md") else base
    return {
        "name": name,
        "content": body,
        "description": description if isinstance(description, str) else None,
    }, diagnostics


async def _load_templates_from_dir(
    env: ExecutionEnv,
    directory: str,
) -> tuple[list[dict[str, Any]], list[PromptTemplateDiagnostic]]:
    templates: list[dict[str, Any]] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    list_result = await env.list_dir(directory)
    if not list_result[0]:
        diagnostics.append(PromptTemplateDiagnostic("list_failed", list_result[1].message, directory))
        return templates, diagnostics
    for entry in sorted(list_result[1], key=lambda e: e.name):
        if entry.kind != "file" or not entry.name.endswith(".md"):
            continue
        template, template_diagnostics = await _load_template_from_file(env, entry.path)
        if template:
            templates.append(template)
        diagnostics.extend(template_diagnostics)
    return templates, diagnostics


async def load_prompt_templates(
    env: ExecutionEnv,
    paths: str | list[str],
) -> dict[str, Any]:
    """从文件或目录加载模板，返回 {promptTemplates, diagnostics}。"""
    templates: list[dict[str, Any]] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in ([paths] if isinstance(paths, str) else paths):
        info = await env.file_info(path)
        if not info[0]:
            if info[1].code != "not_found":
                diagnostics.append(PromptTemplateDiagnostic("file_info_failed", info[1].message, path))
            continue
        if info[1].kind == "directory":
            result, result_diagnostics = await _load_templates_from_dir(env, info[1].path)
            templates.extend(result)
            diagnostics.extend(result_diagnostics)
        elif info[1].kind == "file" and info[1].name.endswith(".md"):
            template, template_diagnostics = await _load_template_from_file(env, info[1].path)
            if template:
                templates.append(template)
            diagnostics.extend(template_diagnostics)
    return {"promptTemplates": templates, "diagnostics": diagnostics}


async def load_sourced_prompt_templates(
    env: ExecutionEnv,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    templates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for input_item in inputs:
        result = await load_prompt_templates(env, input_item["path"])
        for template in result["promptTemplates"]:
            templates.append({"promptTemplate": template, "source": input_item["source"]})
        for diagnostic in result["diagnostics"]:
            diagnostics.append({**vars(diagnostic), "source": input_item["source"]})
    return {"promptTemplates": templates, "diagnostics": diagnostics}
