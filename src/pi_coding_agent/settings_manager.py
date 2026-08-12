"""设置管理器（对齐 TS core/settings-manager.ts）。

职责：
- 双层 settings.json 加载（global ~/.pi/agent + project .pi），深度合并；
- 项目信任感知：未信任项目不加载/不写入项目设置；
- 存储抽象：FileSettingsStorage（带锁）/ InMemorySettingsStorage；
- 迁移旧格式（queueMode / websockets / skills 对象 / retry.maxDelayMs）；
- typed getter/setter：compaction、retry、httpIdleTimeoutMs、terminal、
  images、defaultProjectTrust、systemPrompt 等。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

from ._config import CONFIG_DIR_NAME, get_agent_dir

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

SettingsScope = str  # "global" | "project"


class SettingsStorage:
    """设置存储抽象：with_lock(scope, fn(current) -> next|None)。"""

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None:
        raise NotImplementedError


class FileSettingsStorage(SettingsStorage):
    """文件存储：global ~/.pi/agent/settings.json + project <cwd>/.pi/settings.json。"""

    def __init__(self, cwd: str | Path, agent_dir: str | Path | None = None) -> None:
        self._global_path = (Path(agent_dir) if agent_dir else get_agent_dir()) / "settings.json"
        self._project_path = Path(cwd).expanduser() / CONFIG_DIR_NAME / "settings.json"

    def _path_for(self, scope: SettingsScope) -> Path:
        return self._global_path if scope == "global" else self._project_path

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None:
        path = self._path_for(scope)
        lock = FileLock(str(path) + ".lock", timeout=30)
        lock.acquire()
        try:
            current = None
            if path.is_file():
                current = path.read_text(encoding="utf-8")
            next_content = fn(current)
            if next_content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(next_content, encoding="utf-8")
                tmp.replace(path)
        finally:
            lock.release()


class InMemorySettingsStorage(SettingsStorage):
    """内存存储（测试 / 无文件 I/O 场景）。"""

    def __init__(self) -> None:
        self._global: str | None = None
        self._project: str | None = None
        self._lock = threading.Lock()

    def with_lock(
        self,
        scope: SettingsScope,
        fn: Callable[[str | None], str | None],
    ) -> None:
        with self._lock:
            current = self._global if scope == "global" else self._project
            next_content = fn(current)
            if next_content is not None:
                if scope == "global":
                    self._global = next_content
                else:
                    self._project = next_content


# ---------------------------------------------------------------------------
# 迁移
# ---------------------------------------------------------------------------


def _migrate_settings(raw: dict) -> dict:
    """旧格式迁移（对齐 TS migrateSettings）。"""
    settings = dict(raw)
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings["queueMode"]
        settings.pop("queueMode", None)
    if "transport" not in settings and isinstance(settings.get("websockets"), bool):
        settings["transport"] = "websocket" if settings["websockets"] else "sse"
        settings.pop("websockets", None)
    if (
        "skills" in settings
        and isinstance(settings["skills"], dict)
        and not isinstance(settings["skills"], list)
    ):
        skills_settings = settings["skills"]
        if (
            skills_settings.get("enableSkillCommands") is not None
            and settings.get("enableSkillCommands") is None
        ):
            settings["enableSkillCommands"] = skills_settings["enableSkillCommands"]
        custom = skills_settings.get("customDirectories")
        if isinstance(custom, list) and custom:
            settings["skills"] = custom
        else:
            settings.pop("skills", None)
    retry = settings.get("retry")
    if isinstance(retry, dict):
        provider = retry.get("provider")
        if not isinstance(provider, dict):
            provider = {}
        if retry.get("maxDelayMs") is not None and provider.get("maxRetryDelayMs") is None:
            provider["maxRetryDelayMs"] = retry["maxDelayMs"]
            retry["provider"] = provider
        retry.pop("maxDelayMs", None)
        settings["retry"] = retry
    return settings


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------


class SettingsError:
    def __init__(self, scope: SettingsScope, error: Exception) -> None:
        self.scope = scope
        self.error = error


def _deep_merge_settings(base: dict, override: dict) -> dict:
    """深度合并（数组/标量直接覆盖，嵌套 dict 递归）。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_settings(result[key], value)
        else:
            result[key] = value
    return result


class SettingsManager:
    """typed 设置管理器：global + project 双层，信任感知。"""

    def __init__(
        self,
        storage: SettingsStorage,
        global_settings: dict,
        project_settings: dict,
        *,
        project_trusted: bool = True,
        errors: list[SettingsError] | None = None,
    ) -> None:
        self._storage = storage
        self._global_settings = global_settings
        self._project_settings = project_settings
        self._project_trusted = project_trusted
        self._errors = errors or []
        self._modified_global: set[str] = set()
        self._modified_project: set[str] = set()
        self._settings = _deep_merge_settings(global_settings, project_settings)

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        cwd: str | Path,
        agent_dir: str | Path | None = None,
        *,
        project_trusted: bool = True,
    ) -> "SettingsManager":
        return cls.from_storage(
            FileSettingsStorage(cwd, agent_dir),
            project_trusted=project_trusted,
        )

    @classmethod
    def from_storage(
        cls,
        storage: SettingsStorage,
        *,
        project_trusted: bool = True,
    ) -> "SettingsManager":
        errors: list[SettingsError] = []
        global_settings, global_error = cls._load_scope(storage, "global", True)
        if global_error is not None:
            errors.append(SettingsError("global", global_error))
        project_settings, project_error = cls._load_scope(storage, "project", project_trusted)
        if project_error is not None:
            errors.append(SettingsError("project", project_error))
        return cls(
            storage,
            global_settings,
            project_settings,
            project_trusted=project_trusted,
            errors=errors,
        )

    @classmethod
    def in_memory(
        cls,
        settings: dict | None = None,
        *,
        project_trusted: bool = True,
    ) -> "SettingsManager":
        storage = InMemorySettingsStorage()
        initial = _migrate_settings(dict(settings or {}))
        storage.with_lock(
            "global",
            lambda _current: json.dumps(initial, ensure_ascii=False, indent=2),
        )
        return cls.from_storage(storage, project_trusted=project_trusted)

    @staticmethod
    def _load_scope(
        storage: SettingsStorage,
        scope: SettingsScope,
        project_trusted: bool,
    ) -> tuple[dict, Exception | None]:
        if scope == "project" and not project_trusted:
            return {}, None
        content: str | None = None

        def _read(current: str | None) -> str | None:
            nonlocal content
            content = current
            return None

        try:
            storage.with_lock(scope, _read)
        except Exception as exc:
            return {}, exc
        if not content:
            return {}, None
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("settings.json must be a JSON object")
            return _migrate_settings(parsed), None
        except Exception as exc:
            return {}, exc

    # ------------------------------------------------------------------
    # 基础访问
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """合并后的 settings 快照（兼容现有 load_settings 调用方）。"""
        return dict(self._settings)

    def get_global_settings(self) -> dict:
        return dict(self._global_settings)

    def get_project_settings(self) -> dict:
        return dict(self._project_settings)

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        """切换项目信任并重新加载项目设置。"""
        if self._project_trusted == trusted:
            return
        self._project_trusted = trusted
        self._modified_project.clear()
        if not trusted:
            self._project_settings = {}
        else:
            loaded, error = self._load_scope(self._storage, "project", True)
            if error is not None:
                self._errors.append(SettingsError("project", error))
            self._project_settings = loaded
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)

    def reload(self) -> None:
        """从存储重新加载 global + project。"""
        global_settings, global_error = self._load_scope(self._storage, "global", True)
        if global_error is None:
            self._global_settings = global_settings
        else:
            self._errors.append(SettingsError("global", global_error))
        project_settings, project_error = self._load_scope(
            self._storage, "project", self._project_trusted
        )
        if project_error is None:
            self._project_settings = project_settings
        else:
            self._errors.append(SettingsError("project", project_error))
        self._modified_global.clear()
        self._modified_project.clear()
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)

    def apply_overrides(self, overrides: dict) -> None:
        """在当前合并结果之上应用额外覆盖（不持久化）。"""
        self._settings = _deep_merge_settings(self._settings, dict(overrides))

    def drain_errors(self) -> list[SettingsError]:
        errors = self._errors
        self._errors = []
        return errors

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._settings[key]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _assert_project_trusted(self) -> None:
        if not self._project_trusted:
            raise RuntimeError("Project is not trusted; refusing to write project settings")

    def _set_global(self, key: str, value: Any) -> None:
        self._global_settings[key] = value
        self._modified_global.add(key)
        self._save_global()

    def _set_project(self, key: str, value: Any) -> None:
        self._assert_project_trusted()
        self._project_settings[key] = value
        self._modified_project.add(key)
        self._save_project()

    def set_project_setting(self, key: str, value: Any) -> None:
        """写入项目设置（未信任项目拒绝）。"""
        self._set_project(key, value)

    def _save_global(self) -> None:
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)
        snapshot = dict(self._global_settings)
        modified = set(self._modified_global)

        def _write(current: str | None) -> str:
            base = {}
            if current:
                try:
                    base = _migrate_settings(json.loads(current))
                except (ValueError, json.JSONDecodeError):
                    base = {}
            merged = dict(base)
            for field in modified:
                if field in snapshot:
                    merged[field] = snapshot[field]
                else:
                    merged.pop(field, None)
            return json.dumps(merged, ensure_ascii=False, indent=2)

        try:
            self._storage.with_lock("global", _write)
            self._modified_global.clear()
        except Exception as exc:
            self._errors.append(SettingsError("global", exc))

    def _save_project(self) -> None:
        self._assert_project_trusted()
        self._settings = _deep_merge_settings(self._global_settings, self._project_settings)
        snapshot = dict(self._project_settings)
        modified = set(self._modified_project)

        def _write(current: str | None) -> str:
            base = {}
            if current:
                try:
                    base = _migrate_settings(json.loads(current))
                except (ValueError, json.JSONDecodeError):
                    base = {}
            merged = dict(base)
            for field in modified:
                if field in snapshot:
                    merged[field] = snapshot[field]
                else:
                    merged.pop(field, None)
            return json.dumps(merged, ensure_ascii=False, indent=2)

        try:
            self._storage.with_lock("project", _write)
            self._modified_project.clear()
        except Exception as exc:
            self._errors.append(SettingsError("project", exc))

    def flush(self) -> None:
        """同步存储，写队列已在 setter 中完成；保留接口以对齐 TS。"""

    # ------------------------------------------------------------------
    # typed getters / setters
    # ------------------------------------------------------------------

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._set_global("defaultProvider", provider)

    def set_default_model(self, model_id: str) -> None:
        self._set_global("defaultModel", model_id)

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._global_settings["defaultModel"] = model_id
        self._modified_global.update(("defaultProvider", "defaultModel"))
        self._save_global()

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._set_global("defaultThinkingLevel", level)

    def get_transport(self) -> str:
        return self._settings.get("transport", "auto")

    def set_transport(self, transport: str) -> None:
        self._set_global("transport", transport)

    def get_steering_mode(self) -> str:
        return self._settings.get("steeringMode", "one-at-a-time")

    def set_steering_mode(self, mode: str) -> None:
        self._set_global("steeringMode", mode)

    def get_follow_up_mode(self) -> str:
        return self._settings.get("followUpMode", "one-at-a-time")

    def set_follow_up_mode(self, mode: str) -> None:
        self._set_global("followUpMode", mode)

    def get_theme(self) -> str | None:
        theme = self._settings.get("theme")
        return theme if isinstance(theme, str) else None

    def set_theme(self, theme: str) -> None:
        self._set_global("theme", theme)

    def get_compaction_settings(self) -> dict:
        compaction = self._settings.get("compaction") or {}
        return {
            "enabled": compaction.get("enabled", True),
            "reserveTokens": compaction.get("reserveTokens", 16384),
            "keepRecentTokens": compaction.get("keepRecentTokens", 20000),
        }

    def get_compaction_enabled(self) -> bool:
        return bool(self.get_compaction_settings()["enabled"])

    def set_compaction_enabled(self, enabled: bool) -> None:
        compaction = dict(self._global_settings.get("compaction") or {})
        compaction["enabled"] = bool(enabled)
        self._global_settings["compaction"] = compaction
        self._modified_global.add("compaction")
        self._save_global()

    def get_retry_settings(self) -> dict:
        retry = self._settings.get("retry") or {}
        return {
            "enabled": retry.get("enabled", True),
            "maxRetries": retry.get("maxRetries", 3),
            "baseDelayMs": retry.get("baseDelayMs", 2000),
        }

    def get_provider_retry_settings(self) -> dict:
        """Provider 层请求重试设置（对齐 TS getProviderRetrySettings）。"""
        retry = self._settings.get("retry") or {}
        provider = retry.get("provider") or {}
        return {
            "timeoutMs": provider.get("timeoutMs"),
            "maxRetries": provider.get("maxRetries"),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs", 60000),
        }

    def get_web_socket_connect_timeout_ms(self) -> int | None:
        """WebSocket 连接超时（毫秒）；0 表示禁用（对齐 TS getWebSocketConnectTimeoutMs）。"""
        value = self._settings.get("websocketConnectTimeoutMs")
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
        return None

    def get_branch_summary_settings(self) -> dict:
        """分支摘要设置（对齐 TS getBranchSummarySettings）。"""
        branch_summary = self._settings.get("branchSummary") or {}
        return {
            "reserveTokens": branch_summary.get("reserveTokens", 16384),
            "skipPrompt": branch_summary.get("skipPrompt", False),
        }

    def get_thinking_budgets(self) -> dict | None:
        """自定义思考等级 token 预算（对齐 TS getThinkingBudgets）。"""
        value = self._settings.get("thinkingBudgets")
        return value if isinstance(value, dict) else None

    def get_retry_enabled(self) -> bool:
        return bool(self.get_retry_settings()["enabled"])

    def set_retry_enabled(self, enabled: bool) -> None:
        retry = dict(self._global_settings.get("retry") or {})
        retry["enabled"] = bool(enabled)
        self._global_settings["retry"] = retry
        self._modified_global.add("retry")
        self._save_global()

    def get_http_idle_timeout_ms(self) -> int:
        value = self._settings.get("httpIdleTimeoutMs")
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
        return DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def set_http_idle_timeout_ms(self, timeout_ms: int) -> None:
        if not isinstance(timeout_ms, (int, float)) or timeout_ms < 0:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {timeout_ms}")
        self._set_global("httpIdleTimeoutMs", int(timeout_ms))

    def get_hide_thinking_block(self) -> bool:
        return bool(self._settings.get("hideThinkingBlock", False))

    def set_hide_thinking_block(self, hidden: bool) -> None:
        self._set_global("hideThinkingBlock", bool(hidden))

    def get_show_cache_miss_notices(self) -> bool:
        return bool(self._settings.get("showCacheMissNotices", False))

    def set_show_cache_miss_notices(self, shown: bool) -> None:
        self._set_global("showCacheMissNotices", bool(shown))

    def get_external_editor(self) -> str:
        configured = self._settings.get("externalEditor")
        if isinstance(configured, str) and configured.strip():
            return configured
        import os

        return (
            os.environ.get("VISUAL")
            or os.environ.get("EDITOR")
            or ("notepad" if os.name == "nt" else "nano")
        )

    def get_shell_path(self) -> str | None:
        value = self._settings.get("shellPath")
        return value if isinstance(value, str) and value else None

    def set_shell_path(self, path: str | None) -> None:
        if path is None:
            self._global_settings.pop("shellPath", None)
            self._modified_global.add("shellPath")
        else:
            self._set_global("shellPath", path)

    def get_shell_command_prefix(self) -> str | None:
        value = self._settings.get("shellCommandPrefix")
        return value if isinstance(value, str) and value else None

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        if prefix is None:
            self._global_settings.pop("shellCommandPrefix", None)
            self._modified_global.add("shellCommandPrefix")
        else:
            self._set_global("shellCommandPrefix", prefix)

    def get_quiet_startup(self) -> bool:
        return bool(self._settings.get("quietStartup", False))

    def set_quiet_startup(self, quiet: bool) -> None:
        self._set_global("quietStartup", bool(quiet))

    def get_collapse_changelog(self) -> bool:
        """压缩 changelog 显示（对齐 TS collapseChangelog，默认 False）。"""
        return bool(self._settings.get("collapseChangelog", False))

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._set_global("collapseChangelog", bool(collapse))

    def get_last_changelog_version(self) -> str | None:
        """上次展示 changelog 的版本记录（对齐 TS lastChangelogVersion）。"""
        value = self._global_settings.get("lastChangelogVersion")
        return value if isinstance(value, str) and value else None

    def set_last_changelog_version(self, version: str) -> None:
        self._set_global("lastChangelogVersion", version)
        self._save_global()

    def get_default_project_trust(self) -> str:
        value = self._global_settings.get("defaultProjectTrust")
        return value if value in ("always", "never", "ask") else "ask"

    def set_default_project_trust(self, value: str) -> None:
        if value not in ("always", "never", "ask"):
            raise ValueError(f"Invalid defaultProjectTrust: {value}")
        self._set_global("defaultProjectTrust", value)

    def get_enable_skill_commands(self) -> bool:
        return bool(self._settings.get("enableSkillCommands", True))

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._set_global("enableSkillCommands", bool(enabled))

    def get_show_images(self) -> bool:
        return bool((self._settings.get("terminal") or {}).get("showImages", True))

    def get_ui_mode(self) -> str:
        # 默认 regular（对齐 TS TuiMainScreen）：内容写入主屏并进入终端 scrollback；
        # 通过设置 uiMode=fullscreen 切换为带粘性底部 dock 的 alt-screen 视口。
        return (
            "fullscreen" if self._settings.get("uiMode", "regular") == "fullscreen" else "regular"
        )

    def set_ui_mode(self, mode: str) -> None:
        self._set_global("uiMode", "fullscreen" if mode == "fullscreen" else "regular")

    def get_show_terminal_progress(self) -> bool:
        return bool((self._settings.get("terminal") or {}).get("showTerminalProgress", False))

    def set_show_terminal_progress(self, enabled: bool) -> None:
        terminal = dict(self._global_settings.get("terminal") or {})
        terminal["showTerminalProgress"] = bool(enabled)
        self._global_settings["terminal"] = terminal
        self._modified_global.add("terminal")
        self._save_global()

    def set_show_images(self, show: bool) -> None:
        terminal = dict(self._global_settings.get("terminal") or {})
        terminal["showImages"] = bool(show)
        self._global_settings["terminal"] = terminal
        self._modified_global.add("terminal")
        self._save_global()

    def get_image_width_cells(self) -> int:
        width = (self._settings.get("terminal") or {}).get("imageWidthCells")
        if isinstance(width, (int, float)) and width > 0:
            return max(1, int(width))
        return 60

    def set_image_width_cells(self, width: int) -> None:
        terminal = dict(self._global_settings.get("terminal") or {})
        terminal["imageWidthCells"] = max(1, int(width))
        self._global_settings["terminal"] = terminal
        self._modified_global.add("terminal")
        self._save_global()

    def get_image_auto_resize(self) -> bool:
        return bool((self._settings.get("images") or {}).get("autoResize", True))

    def set_image_auto_resize(self, enabled: bool) -> None:
        images = dict(self._global_settings.get("images") or {})
        images["autoResize"] = bool(enabled)
        self._global_settings["images"] = images
        self._modified_global.add("images")
        self._save_global()

    def get_block_images(self) -> bool:
        return bool((self._settings.get("images") or {}).get("blockImages", False))

    def set_block_images(self, blocked: bool) -> None:
        images = dict(self._global_settings.get("images") or {})
        images["blockImages"] = bool(blocked)
        self._global_settings["images"] = images
        self._modified_global.add("images")
        self._save_global()

    def get_enabled_models(self) -> list[str] | None:
        value = self._settings.get("enabledModels")
        return list(value) if isinstance(value, list) else None

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        if patterns is None:
            self._global_settings.pop("enabledModels", None)
            self._modified_global.add("enabledModels")
            self._save_global()
        else:
            self._set_global("enabledModels", list(patterns))

    def get_double_escape_action(self) -> str:
        return self._settings.get("doubleEscapeAction", "tree")

    def set_double_escape_action(self, action: str) -> None:
        self._set_global("doubleEscapeAction", action)

    def get_tree_filter_mode(self) -> str:
        mode = self._settings.get("treeFilterMode", "default")
        valid = {"default", "no-tools", "user-only", "labeled-only", "all"}
        return mode if mode in valid else "default"

    def set_tree_filter_mode(self, mode: str) -> None:
        self._set_global("treeFilterMode", mode)

    def get_editor_padding_x(self) -> int:
        return int(self._settings.get("editorPaddingX", 0))

    def set_editor_padding_x(self, padding: int) -> None:
        self._set_global("editorPaddingX", max(0, min(3, int(padding))))

    def get_output_pad(self) -> int:
        return 0 if self._settings.get("outputPad") == 0 else 1

    def set_output_pad(self, padding: int) -> None:
        self._set_global("outputPad", 1 if padding else 0)

    def get_autocomplete_max_visible(self) -> int:
        return int(self._settings.get("autocompleteMaxVisible", 5))

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        self._set_global("autocompleteMaxVisible", max(3, min(20, int(max_visible))))

    def get_warnings(self) -> dict:
        warnings = self._settings.get("warnings")
        return dict(warnings) if isinstance(warnings, dict) else {}

    def set_warnings(self, warnings: dict) -> None:
        self._set_global("warnings", dict(warnings))

    def get_session_dir(self) -> str | None:
        value = self._settings.get("sessionDir")
        return value if isinstance(value, str) and value else None

    def get_system_prompt(self) -> str | None:
        value = self._settings.get("systemPrompt")
        return value if isinstance(value, str) and value else None

    def set_system_prompt(self, prompt: str | None) -> None:
        if prompt is None:
            self._global_settings.pop("systemPrompt", None)
            self._modified_global.add("systemPrompt")
            self._save_global()
        else:
            self._set_global("systemPrompt", prompt)

    def get_append_system_prompt(self) -> list[str]:
        value = self._settings.get("appendSystemPrompt")
        if isinstance(value, str) and value:
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return []

    def set_append_system_prompt(self, prompts: list[str]) -> None:
        self._set_global("appendSystemPrompt", list(prompts))

    # ------------------------------------------------------------------
    # 项目级路径设置
    # ------------------------------------------------------------------

    def get_extensions(self) -> list[str]:
        value = self._settings.get("extensions")
        return list(value) if isinstance(value, list) else []

    def set_project_extensions(self, paths: list[str]) -> None:
        self._set_project("extensions", list(paths))

    def get_skills(self) -> list[str]:
        value = self._settings.get("skills")
        return list(value) if isinstance(value, list) else []

    def set_project_skills(self, paths: list[str]) -> None:
        self._set_project("skills", list(paths))

    def get_prompts(self) -> list[str]:
        value = self._settings.get("prompts")
        return list(value) if isinstance(value, list) else []

    def set_project_prompts(self, paths: list[str]) -> None:
        self._set_project("prompts", list(paths))

    def get_themes(self) -> list[str]:
        value = self._settings.get("themes")
        return list(value) if isinstance(value, list) else []

    def set_project_themes(self, paths: list[str]) -> None:
        self._set_project("themes", list(paths))


__all__ = [
    "SettingsManager",
    "SettingsStorage",
    "FileSettingsStorage",
    "InMemorySettingsStorage",
    "SettingsError",
    "DEFAULT_HTTP_IDLE_TIMEOUT_MS",
]
