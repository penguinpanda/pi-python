"""One-time startup migrations aligned with TS packages/coding-agent/src/migrations.ts.

Only the filesystem migrations that are meaningful for the Python CLI are
ported here.  Legacy auth migration remains in ``auth_storage.py`` and is called
separately before these migrations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ._config import get_agent_dir, get_bin_dir


def _session_directory_name(cwd: str) -> str:
    return "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"


def migrate_sessions_from_agent_root() -> None:
    """Move legacy session files from ~/.pi/agent/*.jsonl into cwd-encoded dirs."""
    agent_dir = get_agent_dir()
    try:
        files = [path for path in agent_dir.glob("*.jsonl") if path.is_file()]
    except OSError:
        return

    for file_path in files:
        try:
            first_line = file_path.read_text(encoding="utf-8").split("\n", 1)[0]
            header = json.loads(first_line)
            if header.get("type") != "session" or not header.get("cwd"):
                continue
            correct_dir = agent_dir / "sessions" / _session_directory_name(header["cwd"])
            correct_dir.mkdir(parents=True, exist_ok=True)
            target = correct_dir / file_path.name
            if target.exists():
                continue
            file_path.rename(target)
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def migrate_commands_to_prompts(base_dir: Path) -> bool:
    """Migrate a legacy ``commands/`` directory to ``prompts/``."""
    commands = base_dir / "commands"
    prompts = base_dir / "prompts"
    if commands.exists() and not prompts.exists():
        try:
            commands.rename(prompts)
            return True
        except OSError:
            return False
    return False


def migrate_tools_to_bin() -> None:
    """Move fd/rg binaries from the legacy tools/ directory to bin/."""
    agent_dir = get_agent_dir()
    tools_dir = agent_dir / "tools"
    if not tools_dir.is_dir():
        return
    bin_dir = get_bin_dir(agent_dir)
    for name in ("fd", "rg", "fd.exe", "rg.exe"):
        old_path = tools_dir / name
        new_path = bin_dir / name
        if old_path.exists() and not new_path.exists():
            try:
                bin_dir.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
            except OSError:
                continue


def run_migrations(cwd: str | Path) -> list[str]:
    """Run coding-agent filesystem migrations; returns deprecation warnings."""
    agent_dir = get_agent_dir()
    migrate_sessions_from_agent_root()
    migrate_commands_to_prompts(agent_dir)
    migrate_commands_to_prompts(Path(cwd) / ".pi")
    migrate_tools_to_bin()
    return []


__all__ = [
    "migrate_sessions_from_agent_root",
    "migrate_commands_to_prompts",
    "migrate_tools_to_bin",
    "run_migrations",
]
