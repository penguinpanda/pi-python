"""统一资源加载器（对齐 TS core/resource-loader.ts）。

聚合 skills / prompts / extensions / themes / context-files / system-prompt，
并把各加载器的错误统一为 ResourceDiagnostic 汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pi_tui.theme import BUILTIN_THEMES, Theme, ThemeError, ThemeLoader

from ._config import get_agent_dir
from .extensions import ExtensionLoader
from .prompt_templates import PromptTemplate, PromptTemplateLoader
from .settings_manager import SettingsManager
from .skills import ResourceDiagnostic, Skill, SkillLoader
from .system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
    load_project_context_files,
)


@dataclass(slots=True)
class ResourceLoadResult:
    """一次资源加载的完整结果。"""

    skills: list[Skill] = field(default_factory=list)
    prompts: list[PromptTemplate] = field(default_factory=list)
    extensions: list = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    context_files: list[dict] = field(default_factory=list)
    system_prompt: str | None = None
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)

    def diagnostics_by_type(self, diagnostic_type: str) -> list[ResourceDiagnostic]:
        return [diagnostic for diagnostic in self.diagnostics if diagnostic.type == diagnostic_type]


class DefaultResourceLoader:
    """按目录发现并加载全部资源（项目资源受信任门控）。"""

    def __init__(
        self,
        cwd: str | Path,
        agent_dir: str | Path | None = None,
        *,
        project_trusted: bool = True,
        settings_manager: SettingsManager | None = None,
        selected_tools: list[str] | None = None,
        tool_snippets: dict[str, str] | None = None,
        no_context_files: bool = False,
    ) -> None:
        self._cwd = str(Path(cwd).expanduser().resolve())
        self._agent_dir = Path(agent_dir) if agent_dir else get_agent_dir()
        self._project_trusted = project_trusted
        self._settings_manager = settings_manager or SettingsManager.in_memory(
            project_trusted=project_trusted
        )
        self._selected_tools = selected_tools
        self._tool_snippets = tool_snippets
        self._no_context_files = no_context_files

        self._skill_loader = SkillLoader(
            global_dir=self._agent_dir / "skills",
            project_dir=self._project_dir("skills"),
        )
        self._template_loader = PromptTemplateLoader(
            global_dir=self._agent_dir / "prompts",
            project_dir=self._project_dir("prompts"),
        )
        self._extension_loader = ExtensionLoader(
            global_dir=self._agent_dir / "extensions",
            project_dir=self._project_dir("extensions"),
            cwd=self._cwd,
        )
        self._result = ResourceLoadResult()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _project_dir(self, name: str) -> Path | None:
        if not self._project_trusted:
            return None
        return Path(self._cwd) / ".pi" / name

    def _load_themes(self, diagnostics: list[ResourceDiagnostic]) -> list[Theme]:
        themes: list[Theme] = []
        seen: set[str] = set()
        for name in BUILTIN_THEMES:
            themes.append(Theme(name=name, colors=dict(BUILTIN_THEMES[name])))
            seen.add(name)

        directories = [self._agent_dir / "themes"]
        if self._project_trusted:
            directories.append(Path(self._cwd) / ".pi" / "themes")
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                if not path.is_file():
                    continue
                name = path.stem
                if name in seen:
                    diagnostics.append(
                        ResourceDiagnostic(
                            type="collision",
                            message=f'name "{name}" collision',
                            path=str(path),
                            code="collision",
                        )
                    )
                    continue
                seen.add(name)
                try:
                    themes.append(ThemeLoader(directory).load(name))
                except ThemeError as exc:
                    diagnostics.append(
                        ResourceDiagnostic(
                            type="warning",
                            message=str(exc),
                            path=str(path),
                            code="theme_load_failed",
                        )
                    )
        return themes

    def _build_system_prompt(self, skills: list[Skill]) -> str:
        append_parts = self._settings_manager.get_append_system_prompt()
        options = BuildSystemPromptOptions(
            cwd=self._cwd,
            custom_prompt=self._settings_manager.get_system_prompt(),
            selected_tools=self._selected_tools,
            tool_snippets=self._tool_snippets,
            append_system_prompt="\n".join(append_parts) if append_parts else None,
            context_files=self._result.context_files,
            skills=skills,
        )
        return build_system_prompt(options)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    async def load(self) -> ResourceLoadResult:
        """重新扫描全部资源并返回结果（结果也缓存在实例上）。"""
        diagnostics: list[ResourceDiagnostic] = []

        skill_result = self._skill_loader.load()
        diagnostics.extend(skill_result.diagnostics)

        templates = self._template_loader.load()

        extension_result = await self._extension_loader.load()
        for error in extension_result.errors:
            diagnostics.append(
                ResourceDiagnostic(
                    type="error",
                    message=error.error.replace("\n", " "),
                    path=error.extension_path,
                    code="extension_load_failed",
                )
            )

        themes = self._load_themes(diagnostics)
        context_files = (
            [] if self._no_context_files else load_project_context_files(self._cwd, self._agent_dir)
        )
        self._result = ResourceLoadResult(
            skills=skill_result.skills,
            prompts=templates,
            extensions=extension_result.extensions,
            themes=themes,
            context_files=context_files,
            diagnostics=diagnostics,
        )
        self._result.system_prompt = self._build_system_prompt(skill_result.skills)
        return self._result

    async def reload(self, *, project_trusted: bool | None = None) -> ResourceLoadResult:
        """重新加载；可同时切换项目信任状态。"""
        if project_trusted is not None and project_trusted != self._project_trusted:
            self._project_trusted = project_trusted
            self._settings_manager.set_project_trusted(project_trusted)
            self._skill_loader.set_project_dir(self._project_dir("skills"))
            self._template_loader.set_project_dir(self._project_dir("prompts"))
            self._extension_loader.set_project_dir(self._project_dir("extensions"))
        return await self.load()

    # ------------------------------------------------------------------
    # 访问器
    # ------------------------------------------------------------------

    def get_result(self) -> ResourceLoadResult:
        return self._result

    def get_skills(self) -> list[Skill]:
        return list(self._result.skills)

    def get_prompts(self) -> list[PromptTemplate]:
        return list(self._result.prompts)

    def get_extensions(self) -> list:
        return list(self._result.extensions)

    def get_themes(self) -> list[Theme]:
        return list(self._result.themes)

    def get_context_files(self) -> list[dict]:
        return list(self._result.context_files)

    def get_system_prompt(self) -> str | None:
        return self._result.system_prompt

    def get_diagnostics(self) -> list[ResourceDiagnostic]:
        return list(self._result.diagnostics)


__all__ = [
    "DefaultResourceLoader",
    "ResourceLoadResult",
]
