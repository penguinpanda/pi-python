"""系统提示构建与项目上下文文件加载（对齐 TS core/system-prompt.ts）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ._config import get_agent_dir, get_docs_path, get_examples_path, get_readme_path
from .skills import format_skills_for_prompt

_CONTEXT_FILE_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


@dataclass(slots=True)
class BuildSystemPromptOptions:
    """build_system_prompt 选项（对齐 TS BuildSystemPromptOptions）。"""

    cwd: str
    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] = field(default_factory=list)
    append_system_prompt: str | None = None
    context_files: list[dict] = field(default_factory=list)
    skills: list | None = None


def _tool_snippet(tool) -> str | None:
    """工具的单行说明（描述第一行）。"""
    description = getattr(tool, "description", None)
    if not description:
        return None
    first_line = next(
        (line.strip() for line in str(description).split("\n") if line.strip()),
        "",
    )
    return first_line or None


def tool_snippets_for(tools: list) -> dict[str, str]:
    """为工具列表生成 {name: 单行说明}。"""
    snippets: dict[str, str] = {}
    for tool in tools:
        snippet = _tool_snippet(tool)
        if snippet is not None:
            snippets[tool.name] = snippet
    return snippets


def _load_context_file_from_dir(directory: Path) -> dict | None:
    for filename in _CONTEXT_FILE_NAMES:
        candidate = directory / filename
        if not candidate.is_file():
            continue
        try:
            return {
                "path": str(candidate),
                "content": candidate.read_text(encoding="utf-8"),
            }
        except OSError:
            return None
    return None


@dataclass(slots=True)
class _GitPaths:
    """git 元数据路径（对齐 TS GitPaths）。"""

    repo_dir: Path
    common_git_dir: Path
    head_path: Path


def find_git_paths(cwd: str | Path) -> _GitPaths | None:
    """向上查找 git 元数据（普通仓库 .git 目录 / linked worktree .git 文件）。

    对齐 TS findGitPaths：.git 为文件时读 `gitdir:` 指向的 gitdir，
    再按 gitdir/commondir（缺省为 gitdir 自身）得到公共 git 目录。
    """
    current = Path(cwd).expanduser()
    while True:
        git_path = current / ".git"
        if git_path.exists():
            try:
                if git_path.is_file():
                    content = git_path.read_text(encoding="utf-8", errors="replace").strip()
                    if content.startswith("gitdir: "):
                        git_dir = (current / content[len("gitdir: ") :].strip()).resolve()
                        head_path = git_dir / "HEAD"
                        if not head_path.exists():
                            return None
                        common_dir_path = git_dir / "commondir"
                        if common_dir_path.exists():
                            common_git_dir = (
                                git_dir
                                / common_dir_path.read_text(
                                    encoding="utf-8", errors="replace"
                                ).strip()
                            ).resolve()
                        else:
                            common_git_dir = git_dir
                        return _GitPaths(
                            repo_dir=current.resolve(),
                            common_git_dir=common_git_dir,
                            head_path=head_path,
                        )
                elif git_path.is_dir():
                    head_path = git_path / "HEAD"
                    if not head_path.exists():
                        return None
                    return _GitPaths(
                        repo_dir=current.resolve(),
                        common_git_dir=git_path.resolve(),
                        head_path=head_path,
                    )
            except OSError:
                return None
        if current.parent == current:
            return None
        current = current.parent


def find_shadowed_context_file(cwd: str | Path) -> str | None:
    """返回应被遮蔽的主仓库上下文文件（嵌套 linked worktree 场景）。

    嵌套 linked worktree 的 AGENTS.md/CLAUDE.md 与主仓库是同一份被跟踪
    文件，加载两次会重复；通过 commonGitDir 识别并返回主仓库副本的
    规范化路径（对齐 TS findShadowedContextFile）。
    """
    git_paths = find_git_paths(cwd)
    if git_paths is None:
        return None
    common_git_dir = str(git_paths.common_git_dir)
    worktree_root = str(git_paths.repo_dir)
    main_repo_root = str(Path(common_git_dir).parent)
    if not worktree_root.startswith(main_repo_root + os.sep):
        return None
    if str((Path(main_repo_root) / ".git").resolve()) != common_git_dir:
        return None
    worktree_context = _load_context_file_from_dir(git_paths.repo_dir)
    if worktree_context is None:
        return None
    return str((Path(main_repo_root) / Path(worktree_context["path"]).name).resolve())


def load_project_context_files(cwd: str | Path, agent_dir: str | Path | None = None) -> list[dict]:
    """加载全局 agent 目录 + cwd 祖先链上的 AGENTS.md/CLAUDE.md（去重 + worktree 遮蔽）。"""
    resolved_agent_dir = (Path(agent_dir) if agent_dir else get_agent_dir()).resolve()
    resolved_cwd = Path(cwd).expanduser().resolve()
    context_files: list[dict] = []
    seen: set[str] = set()
    shadowed_context_file = find_shadowed_context_file(resolved_cwd)

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen.add(str(Path(global_context["path"]).resolve()))

    ancestor_files: list[dict] = []
    current = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir(current)
        if context_file is not None:
            canonical = str(Path(context_file["path"]).resolve())
            if canonical != shadowed_context_file and canonical not in seen:
                ancestor_files.insert(0, context_file)
                seen.add(canonical)
        if current.parent == current:
            break
        current = current.parent

    context_files.extend(ancestor_files)
    return context_files


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """构建系统提示：自定义提示 / 工具说明 + 指南 + 上下文文件 + 技能。"""
    custom_prompt = options.custom_prompt
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""
    context_files = options.context_files or []
    skills = options.skills or []
    prompt_cwd = str(Path(options.cwd)).replace("\\", "/")

    if custom_prompt:
        prompt = custom_prompt
        if append_section:
            prompt += append_section
        if context_files:
            prompt += "\n\n<project_context>\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for entry in context_files:
                prompt += (
                    f'<project_instructions path="{entry["path"]}">\n'
                    f"{entry['content']}\n"
                    f"</project_instructions>\n\n"
                )
            prompt += "</project_context>\n"
        selected = options.selected_tools
        custom_prompt_has_read = not selected or "read" in selected
        if custom_prompt_has_read and skills:
            prompt += format_skills_for_prompt(skills)
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    tools = (
        options.selected_tools
        if options.selected_tools is not None
        else ["read", "bash", "edit", "write"]
    )
    snippets = options.tool_snippets or {}
    visible_tools = [name for name in tools if snippets.get(name)]
    tools_list = (
        "\n".join(f"- {name}: {snippets[name]}" for name in visible_tools)
        if visible_tools
        else "(none)"
    )

    guidelines_list: list[str] = []
    guidelines_set: set[str] = set()

    def add_guideline(guideline: str) -> None:
        if guideline not in guidelines_set:
            guidelines_set.add(guideline)
            guidelines_list.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")
    for guideline in options.prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)
    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = (
        "You are an expert coding assistant operating inside pi, a coding agent "
        "harness. You help users by reading files, executing commands, editing "
        "code, and writing new files.\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "In addition to the tools above, you may have access to other custom "
        "tools depending on the project.\n\n"
        f"Guidelines:\n{guidelines}\n"
    )

    # Pi 文档段：固定指向 pi 包自身 README/docs/examples（对齐 TS buildSystemPrompt），
    # 与 cwd 是哪个项目无关；PI_PACKAGE_DIR 可覆盖包目录。
    prompt += (
        "\nPi documentation (read only when the user asks about pi itself, its SDK, "
        "extensions, themes, skills, or TUI):"
        f"\n- Main documentation: {get_readme_path()}"
        f"\n- Additional docs: {get_docs_path()}"
        f"\n- Examples: {get_examples_path()} (extensions, custom tools, SDK)"
        "\n- When asked about: extensions (docs/extensions.md, examples/extensions/), "
        "themes (docs/themes.md), skills (docs/skills.md), prompt templates "
        "(docs/prompt-templates.md), TUI components (docs/tui.md), keybindings "
        "(docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers "
        "(docs/custom-provider.md), adding models (docs/models.md), pi packages "
        "(docs/packages.md), environment variables (docs/environment-variables.md)"
        "\n- When working on pi topics, read the docs and examples, and follow .md "
        "cross-references before implementing"
        "\n- Always read pi .md files completely and follow links to related docs "
        "(e.g., tui.md for TUI API details)"
    )

    if append_section:
        prompt += append_section
    if context_files:
        prompt += "\n\n<project_context>\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for entry in context_files:
            prompt += (
                f'<project_instructions path="{entry["path"]}">\n'
                f"{entry['content']}\n"
                f"</project_instructions>\n\n"
            )
        prompt += "</project_context>\n"
    if has_read and skills:
        prompt += format_skills_for_prompt(skills)
    prompt += f"\nCurrent working directory: {prompt_cwd}"
    return prompt


__all__ = [
    "BuildSystemPromptOptions",
    "build_system_prompt",
    "load_project_context_files",
    "tool_snippets_for",
]
