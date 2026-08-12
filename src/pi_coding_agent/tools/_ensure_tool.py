"""托管工具下载/缓存（对齐 TS utils/tools-manager.ts）。"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

import httpx

from .._config import get_bin_dir as _get_bin_dir

_NETWORK_TIMEOUT_MS = 10_000
_DOWNLOAD_TIMEOUT_MS = 120_000
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _ToolConfig:
    name: str
    repo: str
    binary_name: str
    system_binary_names: tuple[str, ...]
    tag_prefix: str
    get_asset_name: Callable[[str, str, str], str | None]


def _fd_asset(version: str, plat: str, arch: str) -> str | None:
    arch_str = "aarch64" if arch == "arm64" else "x86_64"
    if plat == "darwin":
        return f"fd-v{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        return f"fd-v{version}-{arch_str}-unknown-linux-gnu.tar.gz"
    if plat == "win32":
        return f"fd-v{version}-{arch_str}-pc-windows-msvc.zip"
    return None


def _rg_asset(version: str, plat: str, arch: str) -> str | None:
    arch_str = "aarch64" if arch == "arm64" else "x86_64"
    if plat == "darwin":
        return f"ripgrep-{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        if arch == "arm64":
            return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
        return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
    if plat == "win32":
        return f"ripgrep-{version}-{arch_str}-pc-windows-msvc.zip"
    return None


_TOOL_CONFIGS: dict[str, _ToolConfig] = {
    "fd": _ToolConfig(
        name="fd",
        repo="sharkdp/fd",
        binary_name="fd",
        system_binary_names=("fd", "fdfind"),
        tag_prefix="v",
        get_asset_name=_fd_asset,
    ),
    "rg": _ToolConfig(
        name="ripgrep",
        repo="BurntSushi/ripgrep",
        binary_name="rg",
        system_binary_names=("rg",),
        tag_prefix="",
        get_asset_name=_rg_asset,
    ),
}


def is_offline_mode_enabled() -> bool:
    value = os.environ.get("PI_OFFLINE", "")
    return value.lower() in ("1", "true", "yes")


def get_tool_path(tool: str, bin_dir: str | Path | None = None) -> str | None:
    """返回缓存二进制或系统命令；不存在时返回 None。"""
    config = _TOOL_CONFIGS.get(tool)
    if config is None:
        return None
    override = os.environ.get(f"PI_{tool.upper()}_PATH")
    if override:
        return override
    root = Path(bin_dir) if bin_dir is not None else _get_bin_dir()
    binary = config.binary_name + (".exe" if platform.system() == "Windows" else "")
    local = root / binary
    if local.is_file():
        return str(local)
    for name in config.system_binary_names:
        found = shutil.which(name)
        if found:
            return found
    return None


async def ensure_tool(
    tool: str,
    *,
    silent: bool = False,
    bin_dir: str | Path | None = None,
) -> str | None:
    """确保工具可用；缺失时下载并缓存到 bin_dir。"""
    existing = get_tool_path(tool, bin_dir)
    if existing is not None:
        return existing
    config = _TOOL_CONFIGS.get(tool)
    if config is None:
        return None
    if is_offline_mode_enabled():
        if not silent:
            print(f"{config.name} not found. Offline mode enabled, skipping download.")
        return None
    if platform.system() == "Android":
        if not silent:
            print(f"{config.name} not found. Install with: pkg install {tool}")
        return None
    if not silent:
        print(f"{config.name} not found. Downloading...")
    try:
        path = await _download_tool(config, bin_dir)
        if not silent:
            print(f"{config.name} installed to {path}")
        return path
    except Exception as exc:
        if not silent:
            print(f"Failed to download {config.name}: {exc}")
        return None


async def _download_tool(
    config: _ToolConfig,
    bin_dir: str | Path | None,
) -> str:
    root = Path(bin_dir) if bin_dir is not None else _get_bin_dir()
    root.mkdir(parents=True, exist_ok=True)
    plat, arch = _platform_arch()
    version = await _latest_version(config.repo)
    asset_name = config.get_asset_name(version, plat, arch)
    if asset_name is None:
        raise RuntimeError(f"Unsupported platform: {plat}/{arch}")
    url = (
        f"https://github.com/{config.repo}/releases/download/"
        f"{config.tag_prefix}{version}/{asset_name}"
    )
    archive_path = root / asset_name
    await _download_file(url, archive_path)
    binary_ext = ".exe" if plat == "win32" else ""
    binary_name = config.binary_name + binary_ext
    binary_path = root / binary_name
    try:
        with tempfile.TemporaryDirectory(prefix="pi-tool-", dir=root) as extract_dir:
            _extract_archive(archive_path, Path(extract_dir))
            extracted = _find_binary_recursive(Path(extract_dir), binary_name)
            if extracted is None:
                raise RuntimeError(f"Binary not found in archive: expected {binary_name}")
            binary_path.write_bytes(extracted.read_bytes())
            if plat != "win32":
                binary_path.chmod(0o755)
            (root / f"{binary_name}.version").write_text(
                version + "\n",
                encoding="utf-8",
            )
    finally:
        archive_path.unlink(missing_ok=True)
    return str(binary_path)


async def _latest_version(repo: str) -> str:
    async def fetch() -> str:
        async with httpx.AsyncClient(timeout=_NETWORK_TIMEOUT_MS / 1000) as client:
            response = await client.get(
                f"https://api.github.com/repos/{repo}/releases/latest",
                headers={"User-Agent": "pi-python-coding-agent"},
            )
            response.raise_for_status()
            tag = str(response.json().get("tag_name", ""))
        return tag[1:] if tag.startswith("v") else tag

    return await _with_retry(fetch)


async def _download_file(url: str, dest: Path) -> None:
    async def fetch() -> None:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT_MS / 1000) as client:
            async with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

    await _with_retry(fetch)


async def _with_retry(
    fn: Callable[[], Awaitable[_T]],
    attempts: int = _RETRY_ATTEMPTS,
) -> _T:
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
            if index < len(_RETRY_BACKOFF_SECONDS):
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS[index])
    assert last_error is not None
    raise last_error


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            try:
                archive.extractall(dest_dir, filter="data")
            except TypeError:
                archive.extractall(dest_dir)
        return
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dest_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _find_binary_recursive(root: Path, binary_name: str) -> Path | None:
    for path in root.rglob("*"):
        if path.is_file() and path.name == binary_name:
            return path
    return None


def _platform_arch() -> tuple[str, str]:
    system = platform.system().lower()
    plat = {"linux": "linux", "darwin": "darwin", "windows": "win32"}.get(system, system)
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine
    return plat, arch


__all__ = [
    "ensure_tool",
    "get_tool_path",
    "is_offline_mode_enabled",
]
