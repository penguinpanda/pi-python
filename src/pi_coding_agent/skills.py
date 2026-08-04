"""Skill 发现与加载（对齐 TS core/skills.ts）。

coding-agent 层负责从全局 / 项目 / 显式路径扫描 SKILL.md 并校验；
格式化与调用委托给 pi_agent.skills。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ._config import get_agent_dir
from .frontmatter import parse_frontmatter

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Skill:
    """单个技能（SKILL.md）。"""

    name: str
    description: str
    file_path: str
    base_dir: str
    source: str  # "user" | "project" | "path"
    disable_model_invocation: bool = False


@dataclass(slots=True)
class ResourceDiagnostic:
    """资源加载诊断（warning / collision / error）。"""

    type: str
    message: str
    path: str
    code: str = ""


@dataclass(slots=True)
class LoadSkillsResult:
    skills: list[Skill]
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


# ---------------------------------------------------------------------------
# gitignore 匹配
# ---------------------------------------------------------------------------


def _glob_match(pattern: str, path: str) -> bool:
    """简化 glob（支持 * 与 **）。"""
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", "__DOUBLE__").replace(r"\*", "[^/]*")
    regex = regex.replace("__DOUBLE__", ".*")
    return re.match(f"^{regex}$", path) is not None


class _IgnoreMatcher:
    """gitignore 模式匹配（目录前缀 + glob）。"""

    def __init__(self) -> None:
        self._patterns: list[tuple[str, bool]] = []

    def add(self, patterns: list[str]) -> None:
        for pattern in patterns:
            negated = pattern.startswith("!")
            raw = pattern[1:] if negated else pattern
            self._patterns.append((raw, negated))

    def ignores(self, path: str) -> bool:
        ignored = False
        for raw, negated in self._patterns:
            pattern = raw.rstrip("/")
            if raw.endswith("/"):
                if path == pattern or path.startswith(pattern + "/"):
                    ignored = not negated
            elif _glob_match(pattern, path):
                ignored = not negated
        return ignored


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


def _to_posix(path: str) -> str:
    return path.replace(os.sep, "/")


def _add_ignore_rules(matcher: _IgnoreMatcher, directory: Path, root_dir: Path) -> None:
    relative = directory.relative_to(root_dir)
    prefix = f"{_to_posix(str(relative))}/" if str(relative) != "." else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path = directory / filename
        if not ignore_path.is_file():
            continue
        try:
            content = ignore_path.read_text(encoding="utf-8")
        except OSError:
            continue
        patterns = [
            prefixed
            for line in content.split("\n")
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            matcher.add(patterns)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def _validate_name(name: str) -> list[str]:
    errors: list[str] = []
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


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------


def _load_skill_from_file(
    file_path: Path, source: str
) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []
    try:
        raw_content = file_path.read_text(encoding="utf-8")
        frontmatter, _body = parse_frontmatter(raw_content)
    except OSError as exc:
        diagnostics.append(
            ResourceDiagnostic(
                type="warning", message=str(exc), path=str(file_path), code="read_failed"
            )
        )
        return None, diagnostics

    description = frontmatter.get("description")
    if not isinstance(description, str):
        description = None
    for error in _validate_description(description):
        diagnostics.append(
            ResourceDiagnostic(
                type="warning", message=error, path=str(file_path), code="invalid_metadata"
            )
        )
    frontmatter_name = frontmatter.get("name")
    name = (
        frontmatter_name
        if isinstance(frontmatter_name, str) and frontmatter_name
        else file_path.parent.name
    )
    for error in _validate_name(name):
        diagnostics.append(
            ResourceDiagnostic(
                type="warning", message=error, path=str(file_path), code="invalid_metadata"
            )
        )
    if not description or not description.strip():
        return None, diagnostics
    return (
        Skill(
            name=name,
            description=description,
            file_path=str(file_path),
            base_dir=str(file_path.parent),
            source=source,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _load_skills_from_dir(
    directory: Path,
    source: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher,
    root_dir: Path,
) -> tuple[list[Skill], list[ResourceDiagnostic]]:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    if not directory.is_dir():
        return skills, diagnostics

    _add_ignore_rules(ignore_matcher, directory, root_dir)
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return skills, diagnostics

    # 目录含 SKILL.md → 视为 skill 根，不再递归。
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        if not entry.is_file():
            continue
        rel_path = _to_posix(str(entry.relative_to(root_dir)))
        if ignore_matcher.ignores(rel_path):
            continue
        skill, skill_diagnostics = _load_skill_from_file(entry, source)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)
        return skills, diagnostics

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        rel_path = _to_posix(str(entry.relative_to(root_dir)))
        ignore_path = f"{rel_path}/" if entry.is_dir() else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue
        if entry.is_dir():
            sub_skills, sub_diagnostics = _load_skills_from_dir(
                entry, source, False, ignore_matcher, root_dir
            )
            skills.extend(sub_skills)
            diagnostics.extend(sub_diagnostics)
            continue
        if not entry.is_file() or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, skill_diagnostics = _load_skill_from_file(entry, source)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(skill_diagnostics)
    return skills, diagnostics


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------


class SkillLoader:
    """从全局 / 项目 / 显式路径加载技能。"""

    def __init__(
        self,
        global_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self._global_dir = Path(global_dir) if global_dir else get_agent_dir() / "skills"
        self._project_dir = Path(project_dir) if project_dir else None
        self._skills: dict[str, Skill] = {}
        self._diagnostics: list[ResourceDiagnostic] = []

    def load(self, *, explicit_paths: list[str] | None = None) -> LoadSkillsResult:
        """重新扫描并返回结果（结果也缓存在实例上）。"""
        skill_map: dict[str, Skill] = {}
        real_paths: set[str] = set()
        diagnostics: list[ResourceDiagnostic] = []

        def add(result: LoadSkillsResult) -> None:
            diagnostics.extend(result.diagnostics)
            for skill in result.skills:
                real = str(Path(skill.file_path).resolve())
                if real in real_paths:
                    continue
                existing = skill_map.get(skill.name)
                if existing is not None:
                    diagnostics.append(
                        ResourceDiagnostic(
                            type="collision",
                            message=f'name "{skill.name}" collision',
                            path=skill.file_path,
                            code="collision",
                        )
                    )
                    continue
                skill_map[skill.name] = skill
                real_paths.add(real)

        global_skills, global_diagnostics = _load_skills_from_dir(
            self._global_dir, "user", True, _IgnoreMatcher(), self._global_dir
        )
        add(LoadSkillsResult(skills=global_skills, diagnostics=global_diagnostics))
        if self._project_dir is not None:
            project_skills, project_diagnostics = _load_skills_from_dir(
                self._project_dir, "project", True, _IgnoreMatcher(), self._project_dir
            )
            add(LoadSkillsResult(skills=project_skills, diagnostics=project_diagnostics))

        for raw_path in explicit_paths or []:
            resolved = Path(raw_path).expanduser().resolve()
            if not resolved.exists():
                diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message="skill path does not exist",
                        path=str(resolved),
                        code="path_missing",
                    )
                )
                continue
            if resolved.is_dir():
                path_skills, path_diagnostics = _load_skills_from_dir(
                    resolved, "path", True, _IgnoreMatcher(), resolved
                )
                add(LoadSkillsResult(skills=path_skills, diagnostics=path_diagnostics))
            elif resolved.is_file() and resolved.name.endswith(".md"):
                skill, skill_diagnostics = _load_skill_from_file(resolved, "path")
                if skill is not None:
                    add(LoadSkillsResult(skills=[skill], diagnostics=skill_diagnostics))
                else:
                    diagnostics.extend(skill_diagnostics)

        self._skills = skill_map
        self._diagnostics = diagnostics
        return LoadSkillsResult(skills=list(skill_map.values()), diagnostics=list(diagnostics))

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        """更新项目技能目录（/reload 在信任状态变化后调用）。"""
        self._project_dir = Path(project_dir) if project_dir else None

    def reload(self) -> LoadSkillsResult:
        return self.load()

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def get_diagnostics(self) -> list[ResourceDiagnostic]:
        return list(self._diagnostics)

    def format_for_prompt(self) -> str:
        """格式化为系统提示中的 <available_skills> XML 块。"""
        return format_skills_for_prompt(self.all())

    def format_invocation(self, skill: Skill, additional_instructions: str | None = None) -> str:
        """展开 /skill:name 调用块（复用 pi_agent 格式化）。"""
        try:
            body = parse_frontmatter(Path(skill.file_path).read_text(encoding="utf-8"))[1].strip()
        except OSError:
            body = ""
        from pi_agent.skills import format_skill_invocation

        return format_skill_invocation(
            skill.name,
            skill.description,
            body,
            skill.file_path,
            additional_instructions,
        )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """技能清单 XML（对齐 TS formatSkillsForPrompt；跳过禁用模型调用的技能）。"""
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""

    def escape_xml(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = [
    "MAX_NAME_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "Skill",
    "ResourceDiagnostic",
    "LoadSkillsResult",
    "SkillLoader",
    "format_skills_for_prompt",
]
