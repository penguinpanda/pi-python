"""扩展加载器（5.2）——importlib 动态加载 .py 扩展。

约定：扩展模块导出 `create_extension(api)`（或 `factory(api)`）工厂函数，
同步 / 异步均可。发现规则：目录内直接 `*.py` 文件；子目录含
`index.py` / `pi_extension.py` 视为扩展入口。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .._config import get_agent_dir
from .types import EventBus, Extension, ExtensionAPI, ExtensionError, ExtensionRuntime


@dataclass(slots=True)
class LoadExtensionsResult:
    """加载结果：成功扩展 + 加载错误 + 共享运行时。"""

    extensions: list[Extension]
    errors: list[ExtensionError]
    runtime: ExtensionRuntime


class ExtensionLoader:
    """从全局 / 项目 / 显式路径发现并加载扩展。"""

    def __init__(
        self,
        global_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
        *,
        cwd: str = "",
    ) -> None:
        self._global_dir = Path(global_dir) if global_dir else get_agent_dir() / "extensions"
        self._project_dir = Path(project_dir) if project_dir else None
        self._cwd = cwd

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        """更新项目扩展目录（/reload 在信任状态变化后调用）。"""
        self._project_dir = Path(project_dir) if project_dir else None

    # ------------------------------------------------------------------
    # 发现
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entries(directory: Path) -> list[Path] | None:
        """子目录扩展入口：index.py / pi_extension.py。"""
        for name in ("index.py", "pi_extension.py"):
            candidate = directory / name
            if candidate.is_file():
                return [candidate]
        return None

    def _discover_in_dir(self, directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        discovered: list[Path] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError:
            return discovered
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".py"):
                discovered.append(entry)
            elif entry.is_dir():
                resolved = self._resolve_entries(entry)
                if resolved:
                    discovered.extend(resolved)
        return discovered

    def discover_all(self, explicit_paths: list[str] | None = None) -> list[Path]:
        """发现所有扩展路径（项目 > 全局 > 显式；按解析路径去重）。"""
        paths: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)

        if self._project_dir is not None:
            for path in self._discover_in_dir(self._project_dir):
                add(path)
        for path in self._discover_in_dir(self._global_dir):
            add(path)
        for raw_path in explicit_paths or []:
            resolved = Path(raw_path).expanduser().resolve()
            if resolved.is_dir():
                entries = self._resolve_entries(resolved)
                for entry in entries or self._discover_in_dir(resolved):
                    add(entry)
            else:
                add(resolved)
        return paths

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    async def load_extension(
        self,
        path: Path,
        runtime: ExtensionRuntime,
        event_bus: EventBus | None = None,
    ) -> tuple[Extension | None, ExtensionError | None]:
        """加载单个扩展模块并执行工厂函数。"""
        resolved = path.resolve()
        module_name = f"pi_ext_{uuid.uuid4().hex[:12]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, resolved)
            if spec is None or spec.loader is None:
                return None, ExtensionError(
                    str(path), "load", f"Failed to create module spec: {resolved}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)
        except Exception as exc:
            return None, ExtensionError(str(path), "load", f"Failed to load extension: {exc}")

        factory = getattr(module, "create_extension", None) or getattr(module, "factory", None)
        if not callable(factory):
            return None, ExtensionError(
                str(path),
                "load",
                f"Extension does not export a valid factory function: {path}",
            )

        extension = Extension(
            path=str(path),
            resolved_path=str(resolved),
            source="local",
            base_dir=str(resolved.parent),
        )
        api = ExtensionAPI(
            extension,
            runtime,
            cwd=self._cwd,
            event_bus=event_bus,
        )
        try:
            result = factory(api)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            return None, ExtensionError(
                str(path), "factory", f"Failed to run extension factory: {exc}"
            )
        return extension, None

    async def load(self, explicit_paths: list[str] | None = None) -> LoadExtensionsResult:
        """发现并加载全部扩展，返回结果（错误不中断其它扩展）。"""
        runtime = ExtensionRuntime()
        event_bus = EventBus()
        extensions: list[Extension] = []
        errors: list[ExtensionError] = []
        for path in self.discover_all(explicit_paths):
            extension, error = await self.load_extension(path, runtime, event_bus)
            if error is not None:
                errors.append(error)
            elif extension is not None:
                extensions.append(extension)
        return LoadExtensionsResult(
            extensions=extensions,
            errors=errors,
            runtime=runtime,
        )


__all__ = ["ExtensionLoader", "LoadExtensionsResult"]
