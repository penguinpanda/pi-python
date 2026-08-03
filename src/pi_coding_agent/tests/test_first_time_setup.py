"""首次启动向导测试。"""

from __future__ import annotations

import asyncio

from pi_coding_agent.auth_storage import AuthStorage, FileAuthStorageBackend
from pi_coding_agent._cli import _async_main
from pi_coding_agent.first_time_setup import run_first_time_setup


def _store(tmp_path) -> AuthStorage:
    return AuthStorage(FileAuthStorageBackend(tmp_path / "auth.json"))


def test_wizard_saves_api_key(tmp_path, monkeypatch):
    answers = iter(["1", "sk-test-123"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    store = _store(tmp_path)
    code = asyncio.run(run_first_time_setup(store))
    assert code == 0


def test_wizard_invalid_selection_retries(tmp_path, monkeypatch):
    answers = iter(["99", "2", "sk-deep"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    store = _store(tmp_path)
    code = asyncio.run(run_first_time_setup(store))
    assert code == 0


def test_wizard_empty_key_skips(tmp_path, monkeypatch):
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    store = _store(tmp_path)
    code = asyncio.run(run_first_time_setup(store))
    assert code == 1


def test_cli_setup_runs_inside_event_loop(tmp_path, monkeypatch):
    """回归：--setup 在事件循环内不得调用 asyncio.run（嵌套会抛 RuntimeError）。"""
    from pi_coding_agent import _cli

    store = _store(tmp_path)
    monkeypatch.setattr(_cli, "_auth_store", lambda: store)
    answers = iter(["1", "sk-test-456"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    code = asyncio.run(_async_main(["--setup"]))
    assert code == 0
    assert store.has_key("openai")
