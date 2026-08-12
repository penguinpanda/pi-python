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
AGENT_DIR_NAMES = ("sessions", "prompts", "skills", "extensions", "themes", "tools", "bin")


def _get_home_dir() -> Path:
    """跨平台获取用户主目录。"""
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", str(Path.home())))
    return Path(os.environ.get("HOME", str(Path.home())))


def get_agent_dir() -> Path:
    """全局 agent 目录: ~/.pi/agent/"""
    override = os.environ.get("PI_CODING_AGENT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _get_home_dir() / CONFIG_DIR_NAME / "agent"


def get_sessions_dir(agent_dir: Path | None = None) -> Path:
    """会话存储目录: ~/.pi/agent/sessions/"""
    override = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    if agent_dir is None and override:
        return Path(override).expanduser().resolve()
    return (agent_dir or get_agent_dir()) / "sessions"


def get_skills_dir(agent_dir: Path | None = None) -> Path:
    """全局技能目录: ~/.pi/agent/skills/"""
    return (agent_dir or get_agent_dir()) / "skills"


def get_prompts_dir(agent_dir: Path | None = None) -> Path:
    """全局提示模板目录: ~/.pi/agent/prompts/"""
    return (agent_dir or get_agent_dir()) / "prompts"


def get_themes_dir(agent_dir: Path | None = None) -> Path:
    """用户自定义主题目录: ~/.pi/agent/themes/

    对齐 TS getCustomThemesDir；由 resource_loader 消费（内置 dark/light
    打底，目录内 *.json 主题按名加载）。
    """
    return (agent_dir or get_agent_dir()) / "themes"


def get_tools_dir(agent_dir: Path | None = None) -> Path:
    """工具目录: ~/.pi/agent/tools/

    占位：TS 曾用该目录放自定义工具 / fd、rg 二进制；Python 的工具全部
    内置于 pi_agent / pi_coding_agent，当前未消费该目录。
    """
    return (agent_dir or get_agent_dir()) / "tools"


def get_bin_dir(agent_dir: Path | None = None) -> Path:
    """托管二进制目录: ~/.pi/agent/bin/

    fd/rg 由 `pi_coding_agent.tools._ensure_tool` 下载并缓存到这里。
    """
    return (agent_dir or get_agent_dir()) / "bin"


def get_debug_log_path(agent_dir: Path | None = None) -> Path:
    """调试日志文件: ~/.pi/agent/pi-debug.log

    占位：对齐 TS getDebugLogPath；Python 尚无该日志文件。
    """
    return (agent_dir or get_agent_dir()) / f"{APP_NAME}-debug.log"


def ensure_agent_dirs(agent_dir: Path | None = None) -> None:
    """首次启动补齐 ~/.pi/agent 约定目录（文件仍按需懒创建）。"""
    root = agent_dir or get_agent_dir()
    for name in AGENT_DIR_NAMES:
        (root / name).mkdir(parents=True, exist_ok=True)


def get_settings_path() -> Path:
    """全局设置文件: ~/.pi/agent/settings.json"""
    return get_agent_dir() / "settings.json"


def get_project_settings_path(cwd: str | Path) -> Path:
    """项目设置文件: <cwd>/.pi/settings.json"""
    return Path(cwd) / CONFIG_DIR_NAME / "settings.json"


def get_package_dir() -> Path:
    """pi-coding-agent 包根目录（对齐 TS packages/coding-agent 的 getPackageDir）。

    - PI_PACKAGE_DIR 环境变量可覆盖（用于 Nix/Guix 等 store 路径）。
    - 默认指向本包目录（src/pi_coding_agent），docs/examples/README 随包分发。
    """
    env_dir = os.environ.get("PI_PACKAGE_DIR")
    if env_dir:
        # 不做 resolve()：Windows 形式（如 C:/pi-pkg）在 POSIX 上会被当作
        # 相对路径拼上 cwd，导致输出路径不一致（对齐 TS 直接使用环境变量值）。
        return Path(env_dir)
    return Path(__file__).resolve().parent


def get_changelog_path() -> Path | None:
    """CHANGELOG.md 路径（对齐 TS getChangelogPath）。

    优先包根目录，仓库内开发时回退到仓库根；两者都不存在返回 None。
    """
    candidates = [
        get_package_dir() / "CHANGELOG.md",
        Path(__file__).resolve().parents[2] / "CHANGELOG.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_readme_path() -> Path:
    """pi 包自带 README.md 路径（对齐 TS getReadmePath）。"""
    return get_package_dir() / "README.md"


def get_docs_path() -> Path:
    """pi 包自带 docs 目录路径（对齐 TS getDocsPath）。"""
    return get_package_dir() / "docs"


def get_examples_path() -> Path:
    """pi 包自带 examples 目录路径（对齐 TS getExamplesPath）。"""
    return get_package_dir() / "examples"


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

    关注的设置项:
        - defaultProvider / defaultModel
        - tools.exclude
        - sessionDir
    """
    global_settings = _load_json(get_settings_path())
    project_settings = _load_json(get_project_settings_path(cwd))
    return _deep_merge(global_settings, project_settings)
