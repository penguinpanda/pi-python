"""CHANGELOG 解析与展示（对齐 TS utils/changelog.ts）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ._config import get_agent_dir

_VERSION_RE = re.compile(r"##\s+\[?(\d+)\.(\d+)\.(\d+)\]?")


@dataclass(slots=True, frozen=True)
class ChangelogEntry:
    major: int
    minor: int
    patch: int
    content: str

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def parse_changelog(changelog_path: str | Path) -> list[ChangelogEntry]:
    """从 CHANGELOG.md 解析版本条目（## [x.y.z] 到下一个 ## 或 EOF）。"""
    path = Path(changelog_path)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []

    entries: list[ChangelogEntry] = []
    current: tuple[int, int, int] | None = None
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current is not None and current_lines:
                entries.append(
                    ChangelogEntry(
                        major=current[0],
                        minor=current[1],
                        patch=current[2],
                        content="\n".join(current_lines).strip(),
                    )
                )
            match = _VERSION_RE.match(line)
            if match:
                current = (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
                current_lines = [line]
            else:
                current = None
                current_lines = []
        elif current is not None:
            current_lines.append(line)
    if current is not None and current_lines:
        entries.append(
            ChangelogEntry(
                major=current[0],
                minor=current[1],
                patch=current[2],
                content="\n".join(current_lines).strip(),
            )
        )
    return entries


def compare_versions(v1: ChangelogEntry, v2: ChangelogEntry) -> int:
    """版本比较：v1 < v2 返回负数，相等 0，v1 > v2 正数。"""
    if v1.major != v2.major:
        return v1.major - v2.major
    if v1.minor != v2.minor:
        return v1.minor - v2.minor
    return v1.patch - v2.patch


def get_new_entries(entries: list[ChangelogEntry], last_version: str) -> list[ChangelogEntry]:
    """返回比 last_version 更新的条目。"""
    parts = last_version.split(".")
    last = ChangelogEntry(
        major=int(parts[0]) if len(parts) > 0 else 0,
        minor=int(parts[1]) if len(parts) > 1 else 0,
        patch=int(parts[2]) if len(parts) > 2 else 0,
        content="",
    )
    return [entry for entry in entries if compare_versions(entry, last) > 0]


def find_changelog_path(cwd: str | Path | None = None) -> Path | None:
    """查找 CHANGELOG.md：cwd 祖先链向上搜索，回退全局 agent 目录。"""
    start = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
    current = start
    while True:
        candidate = current / "CHANGELOG.md"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    agent_candidate = get_agent_dir() / "CHANGELOG.md"
    if agent_candidate.is_file():
        return agent_candidate
    return None


def format_changelog(
    entries: list[ChangelogEntry], *, limit: int | None = None
) -> str:
    """把条目渲染为 Markdown 文本（最新在前）。"""
    ordered = sorted(entries, key=lambda entry: entry.version, reverse=True)
    if limit is not None and limit > 0:
        ordered = ordered[:limit]
    if not ordered:
        return "No changelog entries found."
    return "\n\n".join(entry.content for entry in ordered)


__all__ = [
    "ChangelogEntry",
    "parse_changelog",
    "compare_versions",
    "get_new_entries",
    "find_changelog_path",
    "format_changelog",
]
