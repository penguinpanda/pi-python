"""项目信任（对齐 TS core/trust-manager.ts + project-trust.ts）。"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from ._config import get_agent_dir

_RESOURCE_DIRS = ("skills", "prompts", "extensions")


class TrustManager:
    """信任管理器：trust.json 持久化 + 祖先目录继承。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else get_agent_dir() / "trust.json"
        self._data: dict[str, bool | None] = {}
        self.reload()

    def reload(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        self._data = {
            str(key): value
            for key, value in raw.items()
            if isinstance(value, bool) or value is None
        }

    @staticmethod
    def _canonical(path: str) -> str:
        return str(Path(path).expanduser().resolve())

    def is_trusted(self, cwd: str) -> bool | None:
        """祖先目录继承：最近祖先的显式决定优先；无记录返回 None。"""
        current = Path(self._canonical(cwd))
        while True:
            value = self._data.get(str(current))
            if value is not None:
                return value
            if current.parent == current:
                return None
            current = current.parent

    def set_trust(self, cwd: str, trusted: bool) -> None:
        """写入信任决定（原子 + 文件锁）。"""
        canonical = self._canonical(cwd)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._path) + ".lock", timeout=30)
        lock.acquire()
        try:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            raw[canonical] = trusted
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._data = {
                str(key): value
                for key, value in raw.items()
                if isinstance(value, bool) or value is None
            }
        finally:
            lock.release()


def project_has_local_resources(cwd: str) -> bool:
    """项目 `.pi/` 下是否有需要授权的本地资源目录。"""
    pi_dir = Path(cwd).expanduser() / ".pi"
    if not pi_dir.is_dir():
        return False
    return any((pi_dir / name).is_dir() for name in _RESOURCE_DIRS)


async def resolve_project_trusted(
    cwd: str,
    trust_manager: TrustManager,
    settings: dict,
    *,
    ui=None,
    extensions=None,
) -> bool:
    """解析项目信任：

    1. trustOverride 设置 → 直接使用；
    2. 无本地资源 → 自动信任；
    3. 扩展 project_trust 事件 → 扩展决定（undecided 继续）；
    4. trust.json 持久化记录；
    5. defaultProjectTrust 设置（trust / block / ask）；
    6. ask + UI → 交互式确认。
    """
    override = settings.get("trustOverride")
    if override is not None:
        return bool(override)

    if not project_has_local_resources(cwd):
        return True

    if extensions is not None and extensions.has_handlers("project_trust"):
        result = await extensions.emit_project_trust(cwd)
        if result in ("yes", "no"):
            return result == "yes"

    stored = trust_manager.is_trusted(cwd)
    if stored is not None:
        return stored

    default = settings.get("defaultProjectTrust", "ask")
    if default == "trust":
        return True
    if default == "block":
        return False
    if ui is not None:
        confirmed = await ui.confirm(
            "Project trust",
            f"Trust project at {cwd} to load local .pi resources?",
        )
        trust_manager.set_trust(cwd, confirmed)
        return confirmed
    return False


__all__ = [
    "TrustManager",
    "project_has_local_resources",
    "resolve_project_trusted",
]
