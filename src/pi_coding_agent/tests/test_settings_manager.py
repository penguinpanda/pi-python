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
        manager = SettingsManager.in_memory({
            "global": True,
            "nested": {"a": 1},
            "project": 1,
        }, project_trusted=True)
        storage = manager._storage
        assert isinstance(storage, InMemorySettingsStorage)
        manager._storage.with_lock("project", lambda _: json.dumps({
            "project": 2,
            "nested": {"b": 2},
        }))
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
        manager = SettingsManager.create(
            project, agent_dir, project_trusted=False
        )
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
        manager = SettingsManager.in_memory({
            "skills": {
                "enableSkillCommands": False,
                "customDirectories": ["/x/skills"],
            }
        })
        assert manager.get_enable_skill_commands() is False
        assert manager.get_skills() == ["/x/skills"]

    def test_retry_max_delay_migrated(self):
        manager = SettingsManager.in_memory({
            "retry": {"maxDelayMs": 120000}
        })
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
