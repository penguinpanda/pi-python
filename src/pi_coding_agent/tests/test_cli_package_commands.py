"""CLI 包管理/config/模式别名测试（本会话批次新增功能）。"""

from __future__ import annotations

import pytest

from pi_coding_agent._cli import (
    _create_parser,
    _run_config_command,
    _run_package_command,
)


def test_parser_accepts_ts_mode_aliases() -> None:
    parser = _create_parser()
    parsed = parser.parse_args(["--mode", "text", "-p", "hi"])
    assert parsed.mode == "text"
    parsed = parser.parse_args(["--mode", "json", "-p", "hi"])
    assert parsed.mode == "json"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "bogus"])


def test_parser_name_flag() -> None:
    parser = _create_parser()
    parsed = parser.parse_args(["--name", "my session", "-p", "hi"])
    assert parsed.name == "my session"
    parsed = parser.parse_args(["-n", "short", "-p", "hi"])
    assert parsed.name == "short"


@pytest.mark.asyncio
async def test_pi_list_empty(monkeypatch, capsys) -> None:
    import pi_coding_agent._cli as cli

    class _Settings:
        def get_global_settings(self):
            return {}

        def get_project_settings(self):
            return {}

    monkeypatch.setattr(
        cli,
        "SettingsManager",
        type(
            "_SM",
            (),
            {"create": staticmethod(lambda cwd, project_trusted=False: _Settings())},
        ),
    )
    assert await _run_package_command(["list"]) == 0
    assert "No packages installed." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_pi_config_readonly(monkeypatch, capsys, tmp_path) -> None:
    import pi_coding_agent._cli as cli

    called: list[str] = []

    async def _fake_selector(settings_manager, *, cwd, agent_dir, write_scope):
        called.append(write_scope)

    class _Settings:
        def get_global_settings(self):
            return {"packages": ["local:/tmp/fake-ext"]}

        def get_project_settings(self):
            return {}

        def is_project_trusted(self):
            return True

    monkeypatch.setattr(
        cli,
        "SettingsManager",
        type(
            "_SM",
            (),
            {"create": staticmethod(lambda cwd, project_trusted=False: _Settings())},
        ),
    )
    monkeypatch.setattr(cli, "get_agent_dir", lambda: tmp_path / "agent")
    monkeypatch.setattr(
        "pi_coding_agent.config_selector.run_config_selector",
        _fake_selector,
    )
    monkeypatch.chdir(tmp_path)
    assert await _run_config_command([]) == 0
    capsys.readouterr()
    assert called == ["global"]


@pytest.mark.asyncio
async def test_pi_config_local_requires_trust(monkeypatch, capsys, tmp_path) -> None:
    import pi_coding_agent._cli as cli

    class _Settings:
        def get_global_settings(self):
            return {}

        def get_project_settings(self):
            return {}

        def is_project_trusted(self):
            return False

    monkeypatch.setattr(
        cli,
        "SettingsManager",
        type(
            "_SM",
            (),
            {"create": staticmethod(lambda cwd, project_trusted=False: _Settings())},
        ),
    )
    monkeypatch.chdir(tmp_path)
    assert await _run_config_command(["--local"]) == 1
    assert "not trusted" in capsys.readouterr().err
