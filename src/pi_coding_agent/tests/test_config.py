"""双层 settings.json 加载与合并测试（CF-01..CF-05 / P10 根因）。"""

from __future__ import annotations

import json

from pi_coding_agent import _config


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _global_settings(tmp_path):
    return tmp_path / "global" / "settings.json"


class TestLoadSettings:
    async def test_project_overrides_global(self, tmp_path, monkeypatch):
        """项目 defaultModel 覆盖全局；全局其余键保留。"""
        global_path = _global_settings(tmp_path)
        monkeypatch.setattr(_config, "get_settings_path", lambda: global_path)
        _write_json(
            global_path,
            {"defaultProvider": "qwen", "defaultModel": "qwen-plus"},
        )
        project = tmp_path / "proj"
        _write_json(
            project / ".pi" / "settings.json",
            {"defaultModel": "qwen3-vl-flash"},
        )

        merged = _config.load_settings(str(project))

        assert merged["defaultProvider"] == "qwen"
        assert merged["defaultModel"] == "qwen3-vl-flash"

    async def test_deep_merge_nested(self, tmp_path, monkeypatch):
        """嵌套字典深合并：项目只覆盖子键，其余子键保留。"""
        global_path = _global_settings(tmp_path)
        monkeypatch.setattr(_config, "get_settings_path", lambda: global_path)
        _write_json(
            global_path,
            {"compaction": {"enabled": True, "reserveTokens": 128000}},
        )
        project = tmp_path / "proj"
        _write_json(
            project / ".pi" / "settings.json",
            {"compaction": {"reserveTokens": 200000}},
        )

        merged = _config.load_settings(str(project))

        assert merged["compaction"] == {
            "enabled": True,
            "reserveTokens": 200000,
        }

    async def test_cwd_determines_project_path(self, tmp_path, monkeypatch):
        """P10 根因：项目配置按 cwd 解析，不同 cwd 读到不同项目层。"""
        global_path = _global_settings(tmp_path)
        monkeypatch.setattr(_config, "get_settings_path", lambda: global_path)
        _write_json(global_path, {"defaultProvider": "qwen"})
        project_a = tmp_path / "a"
        project_b = tmp_path / "b"
        _write_json(
            project_a / ".pi" / "settings.json",
            {"defaultModel": "model-a"},
        )
        _write_json(
            project_b / ".pi" / "settings.json",
            {"defaultModel": "model-b"},
        )

        assert _config.load_settings(str(project_a))["defaultModel"] == "model-a"
        assert _config.load_settings(str(project_b))["defaultModel"] == "model-b"

    async def test_invalid_project_json_falls_back_to_global(self, tmp_path, monkeypatch):
        """项目 JSON 损坏时不崩溃，回退全局配置。"""
        global_path = _global_settings(tmp_path)
        monkeypatch.setattr(_config, "get_settings_path", lambda: global_path)
        _write_json(global_path, {"defaultProvider": "qwen"})
        project = tmp_path / "proj"
        project_file = project / ".pi" / "settings.json"
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.write_text("{ not json", encoding="utf-8")

        merged = _config.load_settings(str(project))

        assert merged == {"defaultProvider": "qwen"}

    async def test_missing_files_return_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _config,
            "get_settings_path",
            lambda: tmp_path / "nope" / "settings.json",
        )
        assert _config.load_settings(str(tmp_path / "proj")) == {}


class TestPaths:
    def test_global_settings_under_agent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_config, "_get_home_dir", lambda: tmp_path / "home")
        assert _config.get_settings_path() == (
            tmp_path / "home" / ".pi" / "agent" / "settings.json"
        )

    def test_project_settings_under_cwd(self, tmp_path):
        assert _config.get_project_settings_path(tmp_path) == (tmp_path / ".pi" / "settings.json")

    def test_agent_dir_placeholder_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_config, "_get_home_dir", lambda: tmp_path / "home")
        agent_dir = tmp_path / "home" / ".pi" / "agent"
        assert _config.get_themes_dir() == agent_dir / "themes"
        assert _config.get_tools_dir() == agent_dir / "tools"
        assert _config.get_bin_dir() == agent_dir / "bin"
        assert _config.get_debug_log_path() == agent_dir / "pi-debug.log"

    def test_ensure_agent_dirs_creates_convention_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_config, "_get_home_dir", lambda: tmp_path / "home")
        _config.ensure_agent_dirs()
        agent_dir = tmp_path / "home" / ".pi" / "agent"
        for name in _config.AGENT_DIR_NAMES:
            assert (agent_dir / name).is_dir()
