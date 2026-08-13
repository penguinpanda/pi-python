"""`pi config` 交互式资源选择器。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_tui.engine import App, FakeTerminal, Terminal

from ._config import CONFIG_DIR_NAME
from .modes.interactive.components.config_selector import (
    ConfigScope,
    ConfigSelectorComponent,
    ConfigSelectorModel,
    ResourceGroup,
    ResourceItem,
    ResourceScope,
    ResourceType,
)
from .package_manager import PackageManager
from .settings_manager import SettingsManager


RESOURCE_TYPES: tuple[ResourceType, ...] = (
    "extensions",
    "skills",
    "prompts",
    "themes",
)


@dataclass(slots=True)
class _Scope:
    name: ResourceScope
    base_dir: Path


def _posix(path: Path) -> str:
    return path.as_posix()


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _pattern_target(entry: Any) -> str:
    text = str(entry)
    return text[1:] if text and text[0] in "+-!" else text


def _matches_pattern(pattern: str, relative_path: str, absolute_path: str) -> bool:
    target = _pattern_target(pattern).replace("\\", "/")
    return target == relative_path or target == absolute_path.replace("\\", "/")


def _last_matching(entries: list[str], relative_path: str, absolute_path: str) -> str | None:
    for entry in reversed(entries):
        if _matches_pattern(entry, relative_path, absolute_path):
            return entry
    return None


def _enabled_from_patterns(entries: list[str], relative_path: str, absolute_path: str) -> bool:
    if not entries:
        return True
    match = _last_matching(entries, relative_path, absolute_path)
    return match is None or (match[0] not in "-!")


def _iter_extensions(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.name.endswith(".py"):
            result.append(entry)
        elif entry.is_dir() and (
            (entry / "index.py").is_file() or (entry / "pi_extension.py").is_file()
        ):
            result.append(entry)
    return result


def _iter_skills(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for path in sorted(directory.rglob("SKILL.md")):
        result.append(path.parent)
    return result


def _iter_prompts(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix == ".md"]


def _iter_themes(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix == ".json"
    ]


def _iter_resources(directory: Path, resource_type: ResourceType) -> list[Path]:
    if resource_type == "extensions":
        return _iter_extensions(directory)
    if resource_type == "skills":
        return _iter_skills(directory)
    if resource_type == "prompts":
        return _iter_prompts(directory)
    return _iter_themes(directory)


def _display_name(path: Path, resource_type: ResourceType) -> str:
    if resource_type == "skills":
        return path.name
    return path.stem if resource_type in ("prompts", "themes") else path.name


def _settings_paths(
    settings_manager: SettingsManager,
    scope: ResourceScope,
    resource_type: ResourceType,
) -> list[str]:
    if scope == "project":
        value = settings_manager.get_project_settings().get(resource_type, [])
    else:
        value = settings_manager.get_global_settings().get(resource_type, [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _top_level_scope(
    cwd: Path,
    agent_dir: Path,
    scope: ResourceScope,
) -> _Scope:
    return _Scope(scope, agent_dir if scope == "user" else cwd / CONFIG_DIR_NAME)


def _top_level_groups(
    *,
    cwd: Path,
    agent_dir: Path,
    settings_manager: SettingsManager,
    scopes: list[ResourceScope],
) -> list[ResourceGroup]:
    groups: list[ResourceGroup] = []
    for scope in scopes:
        scope_info = _top_level_scope(cwd, agent_dir, scope)
        for resource_type in RESOURCE_TYPES:
            directory = scope_info.base_dir / resource_type
            patterns = _settings_paths(settings_manager, scope, resource_type)
            items: list[ResourceItem] = []
            for path in _iter_resources(directory, resource_type):
                relative_path = _relative(path, scope_info.base_dir)
                absolute_path = _posix(path)
                items.append(
                    ResourceItem(
                        key=f"{resource_type}:{absolute_path}",
                        resource_type=resource_type,
                        path=absolute_path,
                        enabled=_enabled_from_patterns(patterns, relative_path, absolute_path),
                        scope=scope,
                        origin="top-level",
                        source=scope,
                        display_name=_display_name(path, resource_type),
                        base_dir=_posix(scope_info.base_dir),
                    )
                )
            if items:
                groups.append(
                    ResourceGroup(
                        key=f"top-level:{scope}:{resource_type}",
                        label=f"{scope} {resource_type}",
                        scope=scope,
                        origin="top-level",
                        source=scope,
                        items=items,
                    )
                )
    return groups


def _package_entries(
    settings_manager: SettingsManager, scope: ResourceScope
) -> list[dict[str, Any]]:
    settings = (
        settings_manager.get_project_settings()
        if scope == "project"
        else settings_manager.get_global_settings()
    )
    value = settings.get("packages", [])
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, str):
            result.append({"source": entry})
        elif isinstance(entry, dict) and entry.get("source"):
            result.append(entry)
    return result


def _package_scope_base(
    package_manager: PackageManager,
    source: str,
    scope: ResourceScope,
) -> Path:
    return package_manager.installed_path(source, scope)


def _package_groups(
    *,
    cwd: Path,
    agent_dir: Path,
    settings_manager: SettingsManager,
    scopes: list[ResourceScope],
) -> list[ResourceGroup]:
    package_manager = PackageManager(str(cwd), agent_dir, settings_manager=settings_manager)
    groups: list[ResourceGroup] = []
    for scope in scopes:
        for package in _package_entries(settings_manager, scope):
            source = str(package["source"])
            base = _package_scope_base(package_manager, source, scope)
            for resource_type in RESOURCE_TYPES:
                filters = package.get(resource_type)
                patterns = [str(item) for item in filters] if isinstance(filters, list) else []
                directory = base / resource_type
                items: list[ResourceItem] = []
                for path in _iter_resources(directory, resource_type):
                    relative_path = _relative(path, base)
                    absolute_path = _posix(path)
                    items.append(
                        ResourceItem(
                            key=f"{resource_type}:{absolute_path}",
                            resource_type=resource_type,
                            path=absolute_path,
                            enabled=_enabled_from_patterns(
                                patterns,
                                relative_path,
                                absolute_path,
                            ),
                            scope=scope,
                            origin="package",
                            source=source,
                            display_name=_display_name(path, resource_type),
                            base_dir=_posix(base),
                        )
                    )
                if items:
                    groups.append(
                        ResourceGroup(
                            key=f"package:{scope}:{source}:{resource_type}",
                            label=f"{scope} package {source} {resource_type}",
                            scope=scope,
                            origin="package",
                            source=source,
                            items=items,
                        )
                    )
    return groups


def build_config_selector_model(
    *,
    cwd: str | Path,
    agent_dir: str | Path,
    settings_manager: SettingsManager,
    write_scope: ConfigScope,
) -> ConfigSelectorModel:
    cwd_path = Path(cwd).expanduser().resolve()
    agent_path = Path(agent_dir).expanduser().resolve()
    scopes: list[ResourceScope] = ["user"] if write_scope == "global" else ["project", "user"]
    groups = _top_level_groups(
        cwd=cwd_path,
        agent_dir=agent_path,
        settings_manager=settings_manager,
        scopes=scopes,
    )
    groups.extend(
        _package_groups(
            cwd=cwd_path,
            agent_dir=agent_path,
            settings_manager=settings_manager,
            scopes=scopes,
        )
    )
    return ConfigSelectorModel(
        groups=groups,
        cwd=str(cwd_path),
        agent_dir=str(agent_path),
        write_scope=write_scope,
        project_mode_available=settings_manager.is_project_trusted(),
    )


def _update_path_patterns(
    current: list[str],
    relative_path: str,
    absolute_path: str,
    enabled: bool,
) -> list[str]:
    updated = [
        entry for entry in current if not _matches_pattern(entry, relative_path, absolute_path)
    ]
    updated.append(f"{'+' if enabled else '-'}{relative_path}")
    return updated


def _set_top_level_paths(
    settings_manager: SettingsManager,
    item: ResourceItem,
    enabled: bool,
) -> None:
    relative_path = _relative(Path(item.path), Path(item.base_dir))
    absolute_path = item.path
    current = _settings_paths(settings_manager, item.scope, item.resource_type)
    updated = _update_path_patterns(current, relative_path, absolute_path, enabled)
    setter_name = (
        f"set_project_{item.resource_type}"
        if item.scope == "project"
        else f"set_{item.resource_type}"
    )
    setter = getattr(settings_manager, setter_name)
    setter(updated)


def _update_package_entry(
    package: dict[str, Any],
    item: ResourceItem,
    relative_path: str,
    enabled: bool,
) -> dict[str, Any]:
    updated = dict(package)
    current = package.get(item.resource_type)
    entries = [str(entry) for entry in current] if isinstance(current, list) else []
    entries = [entry for entry in entries if _pattern_target(entry) != relative_path]
    entries.append(f"{'+' if enabled else '-'}{relative_path}")
    updated[item.resource_type] = entries
    return updated


def _set_package_paths(
    settings_manager: SettingsManager,
    item: ResourceItem,
    enabled: bool,
) -> None:
    entries = _package_entries(settings_manager, item.scope)
    relative_path = _relative(Path(item.path), Path(item.base_dir))
    updated_entries: list[Any] = []
    for package in entries:
        if package.get("source") != item.source:
            updated_entries.append(package)
            continue
        updated_entries.append(_update_package_entry(package, item, relative_path, enabled))
    if item.scope == "project":
        settings_manager.set_project_setting("packages", updated_entries)
    else:
        settings_manager.set_global_setting("packages", updated_entries)


def persist_resource_toggle(
    settings_manager: SettingsManager,
    item: ResourceItem,
    enabled: bool,
) -> None:
    if item.origin == "top-level":
        _set_top_level_paths(settings_manager, item, enabled)
    else:
        _set_package_paths(settings_manager, item, enabled)


def _default_terminal(size: tuple[int, int] = (80, 24)):
    try:
        return Terminal(size=size)
    except Exception:
        return FakeTerminal(size=size)


class _ConfigSelectorApp(App):
    def __init__(
        self,
        *,
        settings_manager: SettingsManager,
        cwd: Path,
        agent_dir: Path,
        write_scope: ConfigScope,
        terminal,
    ) -> None:
        super().__init__(terminal=terminal, size=terminal.size, ui_mode="fullscreen")
        self._settings_manager = settings_manager
        self._cwd = cwd
        self._agent_dir = agent_dir
        self._write_scope = write_scope

    def _build_component(self) -> ConfigSelectorComponent:
        model = build_config_selector_model(
            cwd=self._cwd,
            agent_dir=self._agent_dir,
            settings_manager=self._settings_manager,
            write_scope=self._write_scope,
        )
        return ConfigSelectorComponent(
            model,
            on_toggle=lambda item, enabled: persist_resource_toggle(
                self._settings_manager,
                item,
                enabled,
            ),
            on_close=self.exit,
            on_exit=self.exit,
            on_switch_scope=self._switch_scope,
        )

    def on_mount(self) -> None:
        component = self._build_component()
        self.screen.mount(component)
        self.focus(component)

    def _switch_scope(self) -> None:
        self._write_scope = "project" if self._write_scope == "global" else "global"
        self.screen.clear()
        self.on_mount()


async def run_config_selector(
    settings_manager: SettingsManager,
    *,
    cwd: str | Path,
    agent_dir: str | Path,
    write_scope: ConfigScope = "global",
    terminal=None,
) -> None:
    app = _ConfigSelectorApp(
        settings_manager=settings_manager,
        cwd=Path(cwd).expanduser().resolve(),
        agent_dir=Path(agent_dir).expanduser().resolve(),
        write_scope=write_scope,
        terminal=terminal or _default_terminal(),
    )
    await app.run_async()


def run_config_selector_sync(
    settings_manager: SettingsManager,
    *,
    cwd: str | Path,
    agent_dir: str | Path,
    write_scope: ConfigScope = "global",
    terminal=None,
) -> None:
    asyncio.run(
        run_config_selector(
            settings_manager,
            cwd=cwd,
            agent_dir=agent_dir,
            write_scope=write_scope,
            terminal=terminal,
        )
    )


__all__ = [
    "build_config_selector_model",
    "persist_resource_toggle",
    "run_config_selector",
    "run_config_selector_sync",
]
