"""包管理测试（install/remove/list/update 核心语义）。"""

from __future__ import annotations

import pytest

from pi_coding_agent.package_manager import (
    PackageManager,
    parse_source,
)


class _FakeSettings:
    def __init__(self, global_settings: dict, project_settings: dict, trusted: bool = True):
        self._global = dict(global_settings)
        self._project = dict(project_settings)
        self._trusted = trusted

    def get_global_settings(self):
        return self._global

    def get_project_settings(self):
        return self._project

    def set_global_setting(self, key, value):
        self._global[key] = value

    def set_project_setting(self, key, value):
        self._project[key] = value

    def is_project_trusted(self):
        return self._trusted


def _manager(tmp_path, settings=None):
    return PackageManager(
        str(tmp_path / "project"),
        agent_dir=tmp_path / "agent",
        settings_manager=settings or _FakeSettings({}, {}),
    )


def test_parse_source() -> None:
    npm = parse_source("npm:pi-extension-example")
    assert (npm.type, npm.name) == ("npm", "pi-extension-example")
    git = parse_source("git:https://github.com/x/y.git@v1")
    assert (git.type, git.name, git.ref) == ("git", "y", "v1")
    # 回归：Windows 本地路径（反斜杠/盘符）的 git 源目录名推导。
    git_win = parse_source("git:C:\\Users\\me\\upstream")
    assert (git_win.type, git_win.name) == ("git", "upstream")
    git_win_trailing = parse_source("git:C:\\Users\\me\\upstream.git\\")
    assert git_win_trailing.name == "upstream"
    local = parse_source("/tmp/my-dir")
    assert (local.type, local.name) == ("local", "my-dir")


def test_parse_source_rejects_unsafe_inputs() -> None:
    """空源 / 路径逃逸包名拒绝（rmtree/命令注入回归）。"""
    with pytest.raises(ValueError):
        parse_source("")
    with pytest.raises(ValueError):
        parse_source("npm:")
    with pytest.raises(ValueError):
        parse_source("git:")
    from pi_coding_agent.package_manager import _assert_safe_package_name

    for bad in (".", "..", "a/b", "a\\b", "a..b"):
        with pytest.raises(ValueError):
            _assert_safe_package_name(bad)


@pytest.mark.asyncio
async def test_git_install_clones_local_repo(tmp_path) -> None:
    """git 源真实 clone（本地仓库，无网络）。"""
    import subprocess

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "main.py").write_text("x", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )

    settings = _FakeSettings({}, {})
    manager = _manager(tmp_path, settings)
    await manager.install_and_persist(f"git:{repo}", local=False)

    installed = manager._installed_path(f"git:{repo}", "user")
    assert (installed / "main.py").exists()
    assert settings.get_global_settings()["packages"] == [f"git:{repo}"]


@pytest.mark.asyncio
async def test_install_missing_local_path_raises(tmp_path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(RuntimeError):
        await manager.install(str(tmp_path / "does-not-exist"), local=False)


@pytest.mark.asyncio
async def test_install_single_file_copy(tmp_path) -> None:
    """local 源为单文件时复制到包目录。"""
    source = tmp_path / "tool.py"
    source.write_text("print(1)", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.install(str(source), local=False)
    installed = manager._installed_path(str(source), "user")
    assert (installed / "tool.py").exists()


@pytest.mark.asyncio
async def test_git_clone_uses_double_dash_separator(tmp_path, monkeypatch) -> None:
    """以 - 开头的 git URL 不当作选项（-- 分隔注入回归）。"""
    import pi_coding_agent.package_manager as pm

    source_dir = tmp_path / "src-ext"
    source_dir.mkdir()
    settings = _FakeSettings({}, {})
    manager = _manager(tmp_path, settings)
    manager.settings_manager = settings

    captured: dict = {}

    async def fake_run(*args, cwd=None):
        captured["args"] = args

    monkeypatch.setattr(pm.PackageManager, "_run", fake_run)
    await manager.install("git:--upload-pack=/bin/sh", local=False)
    args = captured["args"]
    assert "--" in args
    assert args[args.index("--") + 1] == "--upload-pack=/bin/sh"


@pytest.mark.asyncio
async def test_install_local_dir_and_persist(tmp_path) -> None:
    source_dir = tmp_path / "src-ext"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("x", encoding="utf-8")

    settings = _FakeSettings({}, {})
    manager = _manager(tmp_path, settings)
    await manager.install_and_persist(str(source_dir), local=False)

    installed = manager._installed_path(str(source_dir), "user")
    assert (installed / "main.py").exists()
    assert settings.get_global_settings()["packages"] == [str(source_dir)]

    packages = manager.list_configured_packages()
    assert len(packages) == 1
    assert packages[0].scope == "user"
    assert packages[0].installed_path is not None


@pytest.mark.asyncio
async def test_install_project_scope_requires_trust(tmp_path) -> None:
    settings = _FakeSettings({}, {}, trusted=False)
    manager = _manager(tmp_path, settings)
    with pytest.raises(RuntimeError):
        await manager.install("npm:x", local=True)


@pytest.mark.asyncio
async def test_remove_by_identity(tmp_path) -> None:
    settings = _FakeSettings({"packages": ["git:https://github.com/x/y.git"]}, {})
    manager = _manager(tmp_path, settings)
    removed = await manager.remove_and_persist("y")
    assert removed is True
    assert settings.get_global_settings()["packages"] == []


@pytest.mark.asyncio
async def test_remove_missing_returns_false(tmp_path) -> None:
    manager = _manager(tmp_path)
    assert await manager.remove_and_persist("ghost") is False


@pytest.mark.asyncio
async def test_update_reinstalls_configured_sources(tmp_path) -> None:
    source_dir = tmp_path / "src-ext"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("x", encoding="utf-8")
    settings = _FakeSettings({"packages": [str(source_dir)]}, {})
    manager = _manager(tmp_path, settings)

    await manager.update()
    installed = manager._installed_path(str(source_dir), "user")
    assert (installed / "main.py").exists()

    # 指定不存在的源报错
    with pytest.raises(RuntimeError):
        await manager.update("ghost")


@pytest.mark.asyncio
async def test_list_empty(tmp_path) -> None:
    manager = _manager(tmp_path)
    assert manager.list_configured_packages() == []
