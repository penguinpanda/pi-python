"""包管理（对齐 TS core/package-manager.ts 核心子集）。

支持源类型：
- npm:<name>：npm install 到 staging 后复制进资源目录；
- git:<url>[@ref]：git clone；
- 本地目录：复制目录内容。

scope：user（agentDir）/ project（项目 .pi，需信任）。
配置持久化到 settings 的 packages 键（globalSettings / projectSettings）。
"""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ._config import CONFIG_DIR_NAME, get_agent_dir


@dataclass
class ParsedSource:
    type: str  # npm | git | local
    name: str
    spec: str = ""
    ref: str = ""


@dataclass
class ConfiguredPackage:
    source: str
    scope: str  # user | project
    installed_path: str | None
    filtered: bool = False


def parse_source(source: str) -> ParsedSource:
    """解析安装源（对齐 TS parseSource 核心形式）。"""
    value = source.strip()
    if not value:
        raise ValueError("Invalid install source: empty")
    if value.startswith("npm:"):
        name = value[4:].strip()
        if not name:
            raise ValueError("Invalid install source: empty npm package name")
        return ParsedSource(type="npm", name=name, spec=value)
    if value.startswith("git:"):
        rest = value[4:].strip()
        if not rest:
            raise ValueError("Invalid install source: empty git URL")
        ref = ""
        if "@" in rest:
            url, ref = rest.rsplit("@", 1)
            rest = url
        # Windows 本地路径用 \ 分隔：与 / 一并处理。
        name = re.split(r"[/\\]", rest.rstrip("/\\"))[-1].replace(".git", "")
        if not name:
            raise ValueError(f"Invalid install source: {value}")
        return ParsedSource(type="git", name=name, spec=rest, ref=ref)
    name = Path(value).name
    if not name:
        raise ValueError(f"Invalid install source: {value}")
    return ParsedSource(type="local", name=name, spec=value)


def _assert_safe_package_name(name: str) -> None:
    """包名不得为路径分隔/上级目录（防止 rmtree/写入逃逸 packages 目录）。"""
    if not name or name in (".", "..") or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Invalid package name: {name}")


class PackageManager:
    """包安装/移除/更新（对齐 TS DefaultPackageManager 核心子集）。"""

    def __init__(self, cwd: str, agent_dir: Path | None = None, settings_manager=None) -> None:
        self.cwd = Path(cwd).resolve()
        self.agent_dir = agent_dir or get_agent_dir()
        self.settings_manager = settings_manager
        self._progress: list[str] = []

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _base_dir_for_scope(self, scope: str) -> Path:
        return self.agent_dir if scope == "user" else self.cwd / CONFIG_DIR_NAME

    def _packages_dir(self, scope: str) -> Path:
        return self._base_dir_for_scope(scope) / "packages"

    def _installed_path(self, source: str, scope: str) -> Path:
        parsed = parse_source(source)
        return self._packages_dir(scope) / parsed.name

    def installed_path(self, source: str, scope: str) -> Path:
        """配置源对应的安装目录（供资源选择器等只读消费者使用）。"""
        return self._installed_path(source, scope)

    def _assert_project_trusted(self, scope: str) -> None:
        if scope != "project":
            return
        trusted = False
        if self.settings_manager is not None:
            try:
                trusted = self.settings_manager.is_project_trusted()
            except Exception:
                trusted = False
        if not trusted:
            raise RuntimeError(
                "Project is not trusted. Use --approve to modify local package config."
            )

    def _settings_key(self, scope: str) -> str:
        return "packages"

    def _settings_sources(self, scope: str) -> list[str]:
        if self.settings_manager is None:
            return []
        settings = (
            self.settings_manager.get_global_settings()
            if scope == "user"
            else self.settings_manager.get_project_settings()
        )
        packages = settings.get(self._settings_key(scope), [])
        if not isinstance(packages, list):
            return []
        return [item if isinstance(item, str) else str(item.get("source", "")) for item in packages]

    def _set_settings_sources(self, scope: str, sources: list[str]) -> None:
        if self.settings_manager is None:
            raise RuntimeError("No settings manager available")
        if scope == "user":
            self.settings_manager.set_global_setting(self._settings_key(scope), sources)
        else:
            self.settings_manager.set_project_setting(self._settings_key(scope), sources)

    async def _run(self, *args: str, cwd: Path | None = None) -> None:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd or self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            _stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10 * 60)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise RuntimeError(f"Command timed out ({args[0]})") from exc
        if process.returncode != 0:
            raise RuntimeError(f"Command failed ({args[0]}): exit {process.returncode}")

    # ------------------------------------------------------------------
    # 安装
    # ------------------------------------------------------------------

    async def install(self, source: str, *, local: bool = False) -> None:
        scope = "project" if local else "user"
        self._assert_project_trusted(scope)
        parsed = parse_source(source)
        _assert_safe_package_name(parsed.name)
        destination = self._installed_path(source, scope)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if parsed.type == "npm":
            staging = destination.with_name(destination.name + ".staging")
            if staging.exists():
                shutil.rmtree(staging)
            # "--" 分隔：以 "-" 开头的包名不会被当作 npm 选项。
            await self._run("npm", "install", "--prefix", str(staging), "--", parsed.name)
            node_modules = staging / "node_modules" / parsed.name
            if node_modules.exists():
                shutil.copytree(node_modules, destination)
            else:
                shutil.copytree(staging, destination)
            shutil.rmtree(staging, ignore_errors=True)
        elif parsed.type == "git":
            args = ["git", "clone", "--depth", "1"]
            if parsed.ref:
                args += ["--branch", parsed.ref]
            # "--" 分隔：以 "-" 开头的 URL 不会被当作 git 选项。
            args += ["--", parsed.spec, str(destination)]
            await self._run(*args)
        elif parsed.type == "local":
            source_path = Path(parsed.spec).expanduser().resolve()
            if not source_path.exists():
                raise RuntimeError(f"Path does not exist: {source_path}")
            if source_path.is_dir():
                shutil.copytree(source_path, destination)
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination / source_path.name)
        else:
            raise RuntimeError(f"Unsupported install source: {source}")
        self._progress.append(f"Installed {source}")

    async def install_and_persist(self, source: str, *, local: bool = False) -> None:
        await self.install(source, local=local)
        scope = "project" if local else "user"
        sources = self._settings_sources(scope)
        if source not in sources:
            sources.append(source)
            self._set_settings_sources(scope, sources)

    async def remove(self, source: str, *, local: bool = False) -> None:
        """删除已安装资源文件（对齐 TS DefaultPackageManager.remove）。"""
        parsed = parse_source(source)
        if parsed.type == "local":
            return
        scope = "project" if local else "user"
        self._assert_project_trusted(scope)
        destination = self._installed_path(source, scope)
        if destination.exists():
            shutil.rmtree(destination)

    async def remove_and_persist(self, source: str, *, local: bool = False) -> bool:
        scope = "project" if local else "user"
        self._assert_project_trusted(scope)
        sources = self._settings_sources(scope)
        if source not in sources:
            # 允许按包名/身份移除。
            identity = parse_source(source).name
            matched = next(
                (existing for existing in sources if parse_source(existing).name == identity),
                None,
            )
            if matched is None:
                return False
            sources.remove(matched)
            self._set_settings_sources(scope, sources)
            return True
        sources.remove(source)
        self._set_settings_sources(scope, sources)
        return True

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        packages: list[ConfiguredPackage] = []
        for scope in ("user", "project"):
            for source in self._settings_sources(scope):
                installed = self._installed_path(source, scope)
                packages.append(
                    ConfiguredPackage(
                        source=source,
                        scope=scope,
                        installed_path=str(installed) if installed.exists() else None,
                    )
                )
        return packages

    async def update(self, source: str | None = None) -> None:
        """重新安装已配置源（对齐 TS update）。"""
        updated = 0
        for pkg in self.list_configured_packages():
            if source is not None and parse_source(pkg.source).name != parse_source(source).name:
                continue
            await self.install(pkg.source, local=(pkg.scope == "project"))
            updated += 1
        if source is not None and updated == 0:
            raise RuntimeError(f"No matching package found for {source}")
