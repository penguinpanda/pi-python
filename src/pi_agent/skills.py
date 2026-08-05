"""技能系统（Phase 4.4）。

对齐 TS `harness/skills.ts`：

- 从目录递归发现 SKILL.md（以及源目录根的 .md），遵循
  .gitignore / .ignore / .fdignore
- YAML frontmatter 元数据（description、disable-model-invocation）
- 名称约束：64 字符、[a-z0-9-]+，描述上限 1024 字符
- 调用格式化为 <skill> XML 块（属性做 XML 转义防注入）
"""

from __future__ import annotations

import os
import re
from typing import Any

from .env import ExecutionEnv

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")


class SkillDiagnostic:
    def __init__(self, code: str, message: str, path: str, type: str = "warning") -> None:
        self.type = type
        self.code = code
        self.message = message
        self.path = path


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skill_invocation(
    name: str,
    description: str,
    content: str,
    file_path: str,
    additional_instructions: str | None = None,
) -> str:
    """把技能格式化为模型可见的调用块（对齐 TS formatSkillInvocation）。"""
    directory = os.path.dirname(file_path).replace("\\", "/")
    skill_block = (
        f'<skill name="{_xml_escape(name)}" location="{_xml_escape(file_path)}">\n'
        f"References are relative to {_xml_escape(directory)}.\n\n{content}\n</skill>"
    )
    if additional_instructions:
        return f"{skill_block}\n\n{additional_instructions}"
    return skill_block


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """解析 YAML frontmatter 子集（key: value；无 pyyaml 依赖）。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_marker = "\n---"
    end_index = normalized.find(end_marker, 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    frontmatter: dict[str, Any] = {}
    for line in yaml_string.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.lower() == "true":
            frontmatter[key] = True
        elif value.lower() == "false":
            frontmatter[key] = False
        else:
            frontmatter[key] = value
    return frontmatter, body


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )
    return errors


def _dirname(path: str) -> str:
    return os.path.dirname(path)


def _basename(path: str) -> str:
    return os.path.basename(path)


def _relative_path(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


class _IgnoreMatcher:
    """简化 gitignore 匹配（目录前缀 + glob 模式）。"""

    def __init__(self) -> None:
        self.patterns: list[tuple[str, bool]] = []

    def add(self, patterns: list[str]) -> None:
        for pattern in patterns:
            negated = pattern.startswith("!")
            raw = pattern[1:] if negated else pattern
            self.patterns.append((raw, negated))

    def ignores(self, path: str) -> bool:
        ignored = False
        for raw, negated in self.patterns:
            pattern = raw.rstrip("/")
            is_dir_pattern = raw.endswith("/")
            if is_dir_pattern:
                if path == pattern or path.startswith(pattern + "/"):
                    ignored = not negated
            elif _glob_match(pattern, path):
                ignored = not negated
        return ignored


def _glob_match(pattern: str, path: str) -> bool:
    """把简化 glob 转成正则（支持 * 与 **）。"""
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", "__DOUBLE__").replace(r"\*", "[^/]*")
    regex = regex.replace("__DOUBLE__", ".*")
    regex = f"^{regex}$"
    return re.match(regex, path) is not None


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


async def _add_ignore_rules(
    env: ExecutionEnv,
    ignore_matcher: _IgnoreMatcher,
    directory: str,
    root_dir: str,
    diagnostics: list[SkillDiagnostic],
) -> None:
    relative_dir = _relative_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path = f"{directory.rstrip('/')}/{filename}"
        info = await env.file_info(ignore_path)
        if not info[0]:
            if info[1].code != "not_found":
                diagnostics.append(SkillDiagnostic("file_info_failed", str(info[1]), ignore_path))
            continue
        if info[1].kind != "file":
            continue
        content = await env.read_text_file(ignore_path)
        if not content[0]:
            diagnostics.append(SkillDiagnostic("read_failed", str(content[1]), ignore_path))
            continue
        patterns = [
            prefixed
            for line in content[1].split("\n")
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            ignore_matcher.add(patterns)


async def _load_skill_from_file(
    env: ExecutionEnv,
    file_path: str,
) -> tuple[dict[str, Any] | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    raw = await env.read_text_file(file_path)
    if not raw[0]:
        diagnostics.append(SkillDiagnostic("read_failed", str(raw[1]), file_path))
        return None, diagnostics
    try:
        frontmatter, body = _parse_frontmatter(raw[1])
    except Exception as exc:
        diagnostics.append(SkillDiagnostic("parse_failed", str(exc), file_path))
        return None, diagnostics

    skill_dir = _dirname(file_path)
    parent_dir_name = _basename(skill_dir)
    description = frontmatter.get("description")
    if not isinstance(description, str):
        description = None
    for error in _validate_description(description):
        diagnostics.append(SkillDiagnostic("invalid_metadata", error, file_path))
    frontmatter_name = frontmatter.get("name")
    name = frontmatter_name if isinstance(frontmatter_name, str) else parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic("invalid_metadata", error, file_path))
    if not description or not description.strip():
        return None, diagnostics
    return {
        "name": name,
        "description": description,
        "content": body,
        "filePath": file_path,
        "disableModelInvocation": frontmatter.get("disable-model-invocation") is True,
    }, diagnostics


async def _load_skills_from_dir(
    env: ExecutionEnv,
    directory: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher,
    root_dir: str,
) -> tuple[list[dict[str, Any]], list[SkillDiagnostic]]:
    skills: list[dict[str, Any]] = []
    diagnostics: list[SkillDiagnostic] = []

    dir_info = await env.file_info(directory)
    if not dir_info[0]:
        if dir_info[1].code != "not_found":
            diagnostics.append(SkillDiagnostic("file_info_failed", str(dir_info[1]), directory))
        return skills, diagnostics
    if dir_info[1].kind != "directory":
        return skills, diagnostics

    await _add_ignore_rules(env, ignore_matcher, directory, root_dir, diagnostics)
    list_result = await env.list_dir(directory)
    if not list_result[0]:
        diagnostics.append(SkillDiagnostic("list_failed", str(list_result[1]), directory))
        return skills, diagnostics
    entries = list_result[1]

    # SKILL.md 优先
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        if entry.kind != "file":
            continue
        rel_path = _relative_path(root_dir, entry.path)
        if ignore_matcher.ignores(rel_path):
            continue
        skill, skill_diagnostics = await _load_skill_from_file(env, entry.path)
        if skill:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)
        return skills, diagnostics

    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        rel_path = _relative_path(root_dir, entry.path)
        ignore_path = f"{rel_path}/" if entry.kind == "directory" else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue
        if entry.kind == "directory":
            sub_skills, sub_diagnostics = await _load_skills_from_dir(
                env, entry.path, False, ignore_matcher, root_dir
            )
            skills.extend(sub_skills)
            diagnostics.extend(sub_diagnostics)
            continue
        if entry.kind != "file" or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, skill_diagnostics = await _load_skill_from_file(env, entry.path)
        if skill:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)
    return skills, diagnostics


async def load_skills(
    env: ExecutionEnv,
    dirs: str | list[str],
) -> dict[str, Any]:
    """从目录加载技能，返回 {skills, diagnostics}。"""
    skills: list[dict[str, Any]] = []
    diagnostics: list[SkillDiagnostic] = []
    for directory in [dirs] if isinstance(dirs, str) else dirs:
        info = await env.file_info(directory)
        if not info[0]:
            if info[1].code != "not_found":
                diagnostics.append(SkillDiagnostic("file_info_failed", str(info[1]), directory))
            continue
        if info[1].kind != "directory":
            continue
        result, result_diagnostics = await _load_skills_from_dir(
            env, info[1].path, True, _IgnoreMatcher(), info[1].path
        )
        skills.extend(result)
        diagnostics.extend(result_diagnostics)
    return {"skills": skills, "diagnostics": diagnostics}


async def load_sourced_skills(
    env: ExecutionEnv,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """从带来源标记的目录加载技能（来源原样保留）。"""
    skills: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for input_item in inputs:
        result = await load_skills(env, input_item["path"])
        for skill in result["skills"]:
            skills.append({"skill": skill, "source": input_item["source"]})
        for diagnostic in result["diagnostics"]:
            diagnostics.append({**vars(diagnostic), "source": input_item["source"]})
    return {"skills": skills, "diagnostics": diagnostics}
