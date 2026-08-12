"""SettingsManager 测试（对齐 TS settings-manager）。"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.settings_manager import (
    FileSettingsStorage,
    InMemorySettingsStorage,
    SettingsManager,
)


class TestInMemory:
    def test_defaults(self):
        manager = SettingsManager.in_memory()
        assert manager.get_transport() == "auto"
        assert manager.get_compaction_enabled() is True
        assert manager.get_default_project_trust() == "ask"
        assert manager.get_http_idle_timeout_ms() == 300_000
        assert manager.get_show_images() is True
        assert manager.get_image_auto_resize() is True
        assert manager.get_enable_skill_commands() is True

    def test_merged_settings(self):
        manager = SettingsManager.in_memory(
            {
                "global": True,
                "nested": {"a": 1},
                "project": 1,
            },
            project_trusted=True,
        )
        storage = manager._storage
        assert isinstance(storage, InMemorySettingsStorage)
        manager._storage.with_lock(
            "project",
            lambda _: json.dumps(
                {
                    "project": 2,
                    "nested": {"b": 2},
                }
            ),
        )
        manager.reload()
        merged = manager.as_dict()
        assert merged["global"] is True
        assert merged["project"] == 2
        assert merged["nested"] == {"a": 1, "b": 2}

    def test_typed_setters(self):
        manager = SettingsManager.in_memory()
        manager.set_http_idle_timeout_ms(60000)
        assert manager.get_http_idle_timeout_ms() == 60000
        manager.set_compaction_enabled(False)
        assert manager.get_compaction_enabled() is False
        manager.set_default_provider("openai")
        assert manager.get_default_provider() == "openai"
        manager.set_default_project_trust("always")
        assert manager.get_default_project_trust() == "always"

    def test_invalid_http_timeout_raises(self):
        manager = SettingsManager.in_memory()
        with pytest.raises(ValueError):
            manager.set_http_idle_timeout_ms(-1)

    def test_invalid_project_trust_raises(self):
        manager = SettingsManager.in_memory()
        with pytest.raises(ValueError):
            manager.set_default_project_trust("bogus")

    def test_typed_roundtrip_extended(self):
        manager = SettingsManager.in_memory()
        manager.set_default_thinking_level("high")
        assert manager.get_default_thinking_level() == "high"
        manager.set_transport("stdio")
        assert manager.get_transport() == "stdio"
        manager.set_steering_mode("one-at-a-time")
        assert manager.get_steering_mode() == "one-at-a-time"
        manager.set_follow_up_mode("manual")
        assert manager.get_follow_up_mode() == "manual"
        manager.set_theme("dark")
        assert manager.get_theme() == "dark"
        manager.set_hide_thinking_block(True)
        assert manager.get_hide_thinking_block() is True
        manager.set_shell_path("/bin/bash")
        assert manager.get_shell_path() == "/bin/bash"
        manager.set_shell_command_prefix("set -e")
        assert manager.get_shell_command_prefix() == "set -e"
        manager.set_quiet_startup(True)
        assert manager.get_quiet_startup() is True
        manager.set_ui_mode("regular")
        assert manager.get_ui_mode() == "regular"
        manager.set_show_images(False)
        assert manager.get_show_images() is False
        manager.set_image_width_cells(120)
        assert manager.get_image_width_cells() == 120
        manager.set_image_auto_resize(False)
        assert manager.get_image_auto_resize() is False
        manager.set_block_images(True)
        assert manager.get_block_images() is True
        manager.set_enabled_models(["openai/*"])
        assert manager.get_enabled_models() == ["openai/*"]
        manager.set_double_escape_action("abort")
        assert manager.get_double_escape_action() == "abort"
        manager.set_tree_filter_mode("all")
        assert manager.get_tree_filter_mode() == "all"
        manager.set_editor_padding_x(2)
        assert manager.get_editor_padding_x() == 2
        manager.set_output_pad(1)
        assert manager.get_output_pad() == 1
        manager.set_autocomplete_max_visible(10)
        assert manager.get_autocomplete_max_visible() == 10
        manager.set_system_prompt("custom")
        assert manager.get_system_prompt() == "custom"
        manager.set_append_system_prompt(["extra"])
        assert manager.get_append_system_prompt() == ["extra"]
        manager.set_warnings({"duplicateKey": True})
        assert manager.get_warnings() == {"duplicateKey": True}

    def test_invalid_values_clamped(self):
        manager = SettingsManager.in_memory()
        manager.set_ui_mode("bogus")
        assert manager.get_ui_mode() == "regular"
        manager.set_tree_filter_mode("bogus")
        assert manager.get_tree_filter_mode() == "default"


class TestFileStorage:
    def test_global_and_project_merge(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        agent_dir.mkdir(parents=True)
        project.mkdir()
        (agent_dir / "settings.json").write_text(
            json.dumps({"defaultProvider": "global"}), encoding="utf-8"
        )
        (project / ".pi").mkdir()
        (project / ".pi" / "settings.json").write_text(
            json.dumps({"defaultProvider": "project", "defaultModel": "m1"}),
            encoding="utf-8",
        )
        manager = SettingsManager.create(project, agent_dir)
        assert manager.get_default_provider() == "project"
        assert manager.get_default_model() == "m1"

    def test_untrusted_project_ignored(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".pi").mkdir()
        (project / ".pi" / "settings.json").write_text(
            json.dumps({"defaultProvider": "project"}), encoding="utf-8"
        )
        manager = SettingsManager.create(project, agent_dir, project_trusted=False)
        assert manager.get_default_provider() is None

    def test_set_project_trusted_reloads(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".pi").mkdir()
        path = project / ".pi" / "settings.json"
        path.write_text(json.dumps({"defaultModel": "m1"}), encoding="utf-8")
        manager = SettingsManager.create(project, agent_dir, project_trusted=False)
        assert manager.get_default_model() is None
        manager.set_project_trusted(True)
        assert manager.get_default_model() == "m1"
        manager.set_project_trusted(False)
        assert manager.get_default_model() is None

    def test_project_write_refused_when_untrusted(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        project.mkdir()
        manager = SettingsManager.create(project, agent_dir, project_trusted=False)
        with pytest.raises(RuntimeError):
            manager.set_project_extensions(["x"])

    def test_global_setter_persists_without_clobbering(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        project.mkdir()
        settings_path = agent_dir / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"keybindings": {"app.model.select": "ctrl+0"}}),
            encoding="utf-8",
        )
        manager = SettingsManager.create(project, agent_dir)
        manager.set_default_model("qwen-plus")
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data["defaultModel"] == "qwen-plus"
        assert data["keybindings"] == {"app.model.select": "ctrl+0"}

    def test_project_setter_persists(self, tmp_path):
        agent_dir = tmp_path / "agent"
        project = tmp_path / "proj"
        project.mkdir()
        manager = SettingsManager.create(project, agent_dir)
        manager.set_project_skills(["a", "b"])
        path = project / ".pi" / "settings.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["skills"] == ["a", "b"]


class TestMigration:
    def test_queue_mode_migrated(self):
        manager = SettingsManager.in_memory({"queueMode": "all"})
        assert manager.get_steering_mode() == "all"

    def test_websockets_migrated(self):
        manager = SettingsManager.in_memory({"websockets": True})
        assert manager.get_transport() == "websocket"

    def test_skills_object_migrated(self):
        manager = SettingsManager.in_memory(
            {
                "skills": {
                    "enableSkillCommands": False,
                    "customDirectories": ["/x/skills"],
                }
            }
        )
        assert manager.get_enable_skill_commands() is False
        assert manager.get_skills() == ["/x/skills"]

    def test_retry_max_delay_migrated(self):
        manager = SettingsManager.in_memory({"retry": {"maxDelayMs": 120000}})
        retry = manager.as_dict()["retry"]
        assert retry["provider"]["maxRetryDelayMs"] == 120000
        assert "maxDelayMs" not in retry


class TestStorageAbstractions:
    def test_file_storage_lock_roundtrip(self, tmp_path):
        storage = FileSettingsStorage(tmp_path, tmp_path / "agent")
        storage.with_lock("global", lambda _: json.dumps({"a": 1}))
        storage.with_lock("global", lambda current: current)
        path = tmp_path / "agent" / "settings.json"
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_in_memory_storage_roundtrip(self):
        storage = InMemorySettingsStorage()
        storage.with_lock("global", lambda _: json.dumps({"a": 1}))
        read = []
        storage.with_lock("global", lambda current: read.append(current) or None)
        assert json.loads(read[0]) == {"a": 1}


def test_ui_mode_setting(tmp_path) -> None:
    manager = SettingsManager.create(tmp_path, tmp_path / "agent-dir")
    manager.set_global_setting("uiMode", "regular")
    assert manager.as_dict()["uiMode"] == "regular"


def test_tui_display_settings(tmp_path) -> None:
    """mermaidRenderingMode / collapseChangelog / lastChangelogVersion 读写与默认。"""
    manager = SettingsManager.create(tmp_path, tmp_path / "agent-dir")
    # 默认值
    assert manager.get_mermaid_rendering_mode() == "final"
    assert manager.get_collapse_changelog() is False
    assert manager.get_last_changelog_version() is None

    manager.set_mermaid_rendering_mode("streaming")
    manager.set_collapse_changelog(True)
    manager.set_last_changelog_version("0.1.0")
    assert manager.get_mermaid_rendering_mode() == "streaming"
    assert manager.get_collapse_changelog() is True
    assert manager.get_last_changelog_version() == "0.1.0"

    # 非法 mermaid 模式拒绝
    try:
        manager.set_mermaid_rendering_mode("bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # 持久化后可跨实例恢复
    fresh = SettingsManager.create(tmp_path, tmp_path / "agent-dir")
    assert fresh.get_mermaid_rendering_mode() == "streaming"
    assert fresh.get_collapse_changelog() is True
    assert fresh.get_last_changelog_version() == "0.1.0"


def test_ui_mode_in_memory():
    manager = SettingsManager.in_memory({}, project_trusted=True)
    assert manager.get_ui_mode() == "regular"
    manager.set_ui_mode("regular")
    assert manager.get_ui_mode() == "regular"
    manager.set_ui_mode("fullscreen")
    assert manager.get_ui_mode() == "fullscreen"


def test_provider_retry_and_branch_summary_settings(tmp_path):
    """retry.provider / branchSummary / thinkingBudgets / websocket 键访问器。"""
    from pi_coding_agent.settings_manager import SettingsManager

    mgr = SettingsManager.create(tmp_path, tmp_path / "agent-dir")
    mgr._set_global("retry", {"provider": {"maxRetries": 1, "maxRetryDelayMs": 3000}})
    mgr._set_global("branchSummary", {"reserveTokens": 8192, "skipPrompt": True})
    mgr._set_global("thinkingBudgets", {"high": 32000})
    mgr._set_global("websocketConnectTimeoutMs", 5000)
    mgr._save_global()

    retry = mgr.get_provider_retry_settings()
    assert retry == {"timeoutMs": None, "maxRetries": 1, "maxRetryDelayMs": 3000}
    assert mgr.get_branch_summary_settings() == {"reserveTokens": 8192, "skipPrompt": True}
    assert mgr.get_thinking_budgets() == {"high": 32000}
    assert mgr.get_web_socket_connect_timeout_ms() == 5000

    # 持久化后可跨实例恢复
    fresh = SettingsManager.create(tmp_path, tmp_path / "agent-dir")
    assert fresh.get_provider_retry_settings()["maxRetryDelayMs"] == 3000
    assert fresh.get_branch_summary_settings() == {"reserveTokens": 8192, "skipPrompt": True}
    assert fresh.get_thinking_budgets() == {"high": 32000}
    assert fresh.get_web_socket_connect_timeout_ms() == 5000

    # 空实例：全部默认
    empty = SettingsManager.create(tmp_path / "empty", tmp_path / "empty-agent-dir")
    assert empty.get_provider_retry_settings()["maxRetryDelayMs"] == 60000
    assert empty.get_branch_summary_settings() == {"reserveTokens": 16384, "skipPrompt": False}
    assert empty.get_thinking_budgets() is None
    assert empty.get_web_socket_connect_timeout_ms() is None
