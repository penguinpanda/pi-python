"""项目信任（对齐 TS core/trust-manager.ts + project-trust.ts）。"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from ._config import get_agent_dir

_RESOURCE_DIRS = ("skills", "prompts", "extensions")
_TRUST_REQUIRING_ENTRIES = (
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
)


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
        entry = self.get_trust_entry(cwd)
        return entry["decision"] if entry is not None else None

    def get_trust_entry(self, cwd: str) -> dict | None:
        """返回最近祖先的显式决定 {path, decision}；无记录返回 None。"""
        current = Path(self._canonical(cwd))
        while True:
            value = self._data.get(str(current))
            if value is not None:
                return {"path": str(current), "decision": value}
            if current.parent == current:
                return None
            current = current.parent

    def set_trust(self, cwd: str, trusted: bool) -> None:
        """写入信任决定（原子 + 文件锁）。"""
        self.set_many([{"path": cwd, "decision": trusted}])

    def clear_trust(self, cwd: str) -> None:
        """删除指定目录的信任决定（继承回退到祖先）。"""
        self.set_many([{"path": cwd, "decision": None}])

    def set_many(self, updates: list[dict]) -> None:
        """批量写入信任决定（原子 + 文件锁）。

        updates 元素: {"path": str, "decision": bool | None}；decision 为
        None 时删除该路径的记录（对齐 TS setMany）。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._path) + ".lock", timeout=30)
        lock.acquire()
        try:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            for update in updates:
                path = self._canonical(update["path"])
                decision = update.get("decision")
                if decision is None:
                    raw.pop(path, None)
                else:
                    raw[path] = bool(decision)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self.reload()
        finally:
            lock.release()


def project_has_local_resources(cwd: str) -> bool:
    """项目 `.pi/` 下是否有需要授权的本地资源目录。"""
    pi_dir = Path(cwd).expanduser() / ".pi"
    if not pi_dir.is_dir():
        return False
    return any((pi_dir / name).is_dir() for name in _RESOURCE_DIRS)


def has_trust_requiring_project_resources(cwd: str) -> bool:
    """项目是否存在需要信任门控的资源（对齐 TS hasTrustRequiringProjectResources）。

    覆盖 `.pi` 下的 settings/extensions/skills/prompts/themes/SYSTEM.md/
    APPEND_SYSTEM.md，以及 cwd 祖先链上的 `.agents/skills`
    （用户全局 `~/.agents/skills` 视为受信用户资源，排除）。
    """
    user_home = Path.home().resolve()
    user_agents_skills = user_home / ".agents" / "skills"
    current = Path(cwd).expanduser().resolve()

    pi_dir = current / ".pi"
    if any((pi_dir / name).exists() for name in _TRUST_REQUIRING_ENTRIES):
        return True

    while True:
        agents_skills = current / ".agents" / "skills"
        if agents_skills != user_agents_skills and agents_skills.exists():
            return True
        if current == user_home or current.parent == current:
            return False
        current = current.parent


def get_project_trust_options(cwd: str, *, include_session_only: bool = False) -> list[dict]:
    """项目信任选项（对齐 TS getProjectTrustOptions）。"""
    canonical = str(Path(cwd).expanduser().resolve())
    parent = str(Path(canonical).parent)
    options: list[dict] = [
        {
            "label": "Trust",
            "trusted": True,
            "updates": [{"path": canonical, "decision": True}],
            "savedPath": canonical,
        },
    ]
    if parent != canonical:
        options.append(
            {
                "label": f"Trust parent folder ({parent})",
                "trusted": True,
                "updates": [
                    {"path": parent, "decision": True},
                    {"path": canonical, "decision": None},
                ],
                "savedPath": parent,
            }
        )
    if include_session_only:
        options.append(
            {
                "label": "Trust (this session only)",
                "trusted": True,
                "updates": [],
                "savedPath": None,
            }
        )
    options.append(
        {
            "label": "Do not trust",
            "trusted": False,
            "updates": [{"path": canonical, "decision": False}],
            "savedPath": canonical,
        }
    )
    if include_session_only:
        options.append(
            {
                "label": "Do not trust (this session only)",
                "trusted": False,
                "updates": [],
                "savedPath": None,
            }
        )
    return options


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

    if not has_trust_requiring_project_resources(cwd):
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
    "has_trust_requiring_project_resources",
    "get_project_trust_options",
    "resolve_project_trusted",
]
