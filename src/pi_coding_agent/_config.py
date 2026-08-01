"""
pi-coding-agent 配置模块

路径常量 + 双层 settings.json 加载。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APP_NAME = "pi"
CONFIG_DIR_NAME = ".pi"


def _get_home_dir() -> Path:
    """跨平台获取用户主目录。"""
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", str(Path.home())))
    return Path(os.environ.get("HOME", str(Path.home())))


def get_agent_dir() -> Path:
    """全局 agent 目录: ~/.pi/agent/"""
    return _get_home_dir() / CONFIG_DIR_NAME / "agent"


def get_sessions_dir(agent_dir: Path | None = None) -> Path:
    """会话存储目录: ~/.pi/agent/sessions/"""
    return (agent_dir or get_agent_dir()) / "sessions"


def get_settings_path() -> Path:
    """全局设置文件: ~/.pi/agent/settings.json"""
    return get_agent_dir() / "settings.json"


def get_project_settings_path(cwd: str | Path) -> Path:
    """项目设置文件: <cwd>/.pi/settings.json"""
    return Path(cwd) / CONFIG_DIR_NAME / "settings.json"


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_json(path: Path) -> dict:
    """安全加载 JSON 文件，不存在则返回空 dict。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_settings(cwd: str | Path) -> dict:
    """双层合并加载配置: 项目覆盖全局。

    最小核心关注的设置项:
        - defaultProvider / defaultModel
        - tools.exclude
        - sessionDir
    """
    global_settings = _load_json(get_settings_path())
    project_settings = _load_json(get_project_settings_path(cwd))
    return _deep_merge(global_settings, project_settings)
