"""提示模板加载与展开（对齐 TS core/prompt-templates.ts）。

coding-agent 层负责发现/加载 .md 模板并解析参数；
参数替换委托给 pi_agent.prompt_templates.substitute_args。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pi_agent.prompt_templates import substitute_args

from ._config import get_agent_dir
from .frontmatter import parse_frontmatter

_DESCRIPTION_MAX = 60


@dataclass(slots=True, frozen=True)
class PromptTemplate:
    """单个提示模板（.md 文件）。"""

    name: str
    description: str
    argument_hint: str | None
    content: str
    file_path: str
    source: str  # "user" | "project" | "path"


def parse_command_args(args_string: str) -> list[str]:
    """解析命令参数（支持引号包裹，对齐 TS parseCommandArgs）。"""
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def _load_template_from_file(file_path: Path, source: str) -> PromptTemplate | None:
    try:
        raw_content = file_path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw_content)
    except OSError:
        return None

    name = file_path.stem
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        first_line = next((line for line in body.split("\n") if line.strip()), "")
        description = first_line
        if len(first_line) > _DESCRIPTION_MAX:
            description = first_line[:_DESCRIPTION_MAX] + "..."
    argument_hint = frontmatter.get("argument-hint")
    return PromptTemplate(
        name=name,
        description=description,
        argument_hint=argument_hint if isinstance(argument_hint, str) else None,
        content=body,
        file_path=str(file_path),
        source=source,
    )


def _load_templates_from_dir(directory: Path, source: str) -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    if not directory.is_dir():
        return templates
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return templates
    for entry in entries:
        if entry.is_file() and entry.name.endswith(".md"):
            template = _load_template_from_file(entry, source)
            if template is not None:
                templates.append(template)
    return templates


class PromptTemplateLoader:
    """从全局 / 项目 / 显式路径加载提示模板。"""

    def __init__(
        self,
        global_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self._global_dir = Path(global_dir) if global_dir else get_agent_dir() / "prompts"
        self._project_dir = Path(project_dir) if project_dir else None
        self._templates: dict[str, PromptTemplate] = {}

    def load(self, *, explicit_paths: list[str] | None = None) -> list[PromptTemplate]:
        """重新扫描并返回模板列表（结果也缓存在实例上）。"""
        templates: dict[str, PromptTemplate] = {}

        def add(loaded: list[PromptTemplate]) -> None:
            for template in loaded:
                templates.setdefault(template.name, template)

        add(_load_templates_from_dir(self._global_dir, "user"))
        if self._project_dir is not None:
            add(_load_templates_from_dir(self._project_dir, "project"))

        for raw_path in explicit_paths or []:
            resolved = Path(raw_path).expanduser().resolve()
            if not resolved.exists():
                continue
            if resolved.is_dir():
                add(_load_templates_from_dir(resolved, "path"))
            elif resolved.is_file() and resolved.name.endswith(".md"):
                template = _load_template_from_file(resolved, "path")
                if template is not None:
                    templates.setdefault(template.name, template)

        self._templates = templates
        return list(templates.values())

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        """更新项目模板目录（/reload 在信任状态变化后调用）。"""
        self._project_dir = Path(project_dir) if project_dir else None

    def reload(self) -> list[PromptTemplate]:
        return self.load()

    def get(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def all(self) -> list[PromptTemplate]:
        return list(self._templates.values())

    def expand_template(self, template: PromptTemplate, args_string: str) -> str:
        """按参数展开模板内容。"""
        args = parse_command_args(args_string)
        return substitute_args(template.content, args)

    def expand(self, text: str) -> str:
        """`/name [args]` → 展开内容；未匹配时原样返回。"""
        if not text.startswith("/"):
            return text
        match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", text)
        if not match:
            return text
        template_name = match.group(1)
        args_string = match.group(2) or ""
        template = self._templates.get(template_name)
        if template is None:
            return text
        return self.expand_template(template, args_string)


__all__ = [
    "PromptTemplate",
    "PromptTemplateLoader",
    "parse_command_args",
]
