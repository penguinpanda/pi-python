"""env.py 补充测试：路径解析、错误映射、超时与清理。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_agent.env import (
    ExecutionError,
    FileError,
    PythonExecutionEnv,
    ShellExecOptions,
    err,
    get_or_throw,
    ok,
    to_execution_error,
    to_file_error,
)


def _env(tmp_path: Path) -> PythonExecutionEnv:
    return PythonExecutionEnv(str(tmp_path))


def test_result_helpers_and_error_mapping(tmp_path: Path) -> None:
    assert ok(1) == (True, 1)
    file_error = FileError("not_found", "missing", str(tmp_path))
    failed = err(file_error)
    assert failed[0] is False
    assert failed[1].code == "not_found"
    assert failed[1].path == str(tmp_path)
    assert get_or_throw(ok(1)) == 1
    with pytest.raises(FileError):
        get_or_throw(err(FileError("not_found", "missing", str(tmp_path))))

    assert to_file_error(FileNotFoundError("x")).code == "not_found"
    assert to_file_error(PermissionError("x")).code == "permission_denied"
    assert to_file_error(NotADirectoryError("x")).code == "not_directory"
    assert to_file_error(IsADirectoryError("x")).code == "is_directory"
    assert to_file_error(ValueError("x")).code == "invalid"
    assert to_file_error(RuntimeError("x")).code == "unknown"
    assert to_execution_error(ExecutionError("timeout", "t")).code == "timeout"
    assert to_execution_error(RuntimeError("x")).code == "unknown"


@pytest.mark.asyncio
async def test_resolve_path_home_and_file_url(monkeypatch, tmp_path: Path) -> None:
    env = _env(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    tilde = await env.absolute_path("~/file.txt")
    assert tilde[0] is True
    assert tilde[1] == str(tmp_path / "home" / "file.txt")

    file_url = await env.absolute_path(f"file://{tmp_path / 'a b' / 'x.txt'}")
    assert file_url[0] is True
    assert file_url[1] == str(tmp_path / "a b" / "x.txt")


@pytest.mark.asyncio
async def test_read_text_lines_max_lines_zero(tmp_path: Path) -> None:
    env = _env(tmp_path)
    path = tmp_path / "data.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    result = await env.read_text_lines(str(path), {"maxLines": 0})
    assert result == (True, [])


@pytest.mark.asyncio
async def test_create_temp_file_and_remove_without_force(tmp_path: Path) -> None:
    env = _env(tmp_path)
    created = await env.create_temp_file({"prefix": "pfx-", "suffix": ".log"})
    assert created[0] is True
    created_path = Path(created[1])
    assert created_path.name.startswith("pfx-")
    assert created_path.name.endswith(".log")

    missing = tmp_path / "missing.txt"
    removed = await env.remove(str(missing), {"force": False})
    assert removed[0] is False
    assert removed[1].code == "not_found"


@pytest.mark.asyncio
async def test_exec_rejects_invalid_timeout_and_pre_abort(tmp_path: Path) -> None:
    env = _env(tmp_path)
    bad_timeout = await env.exec("echo x", ShellExecOptions(timeout=0))
    assert bad_timeout[0] is False
    assert bad_timeout[1].code == "timeout"

    signal = asyncio.Event()
    signal.set()
    aborted = await env.exec("echo x", ShellExecOptions(abort_signal=signal))
    assert aborted[0] is False
    assert aborted[1].code == "aborted"


@pytest.mark.asyncio
async def test_exec_stdout_callback_error(tmp_path: Path) -> None:
    env = _env(tmp_path)

    def fail_callback(_chunk: str) -> None:
        raise RuntimeError("callback boom")

    result = await env.exec("echo hi", ShellExecOptions(on_stdout=fail_callback))
    assert result[0] is False
    assert result[1].code == "unknown"


@pytest.mark.asyncio
async def test_cleanup_kills_active_process(tmp_path: Path) -> None:
    env = _env(tmp_path)
    process = await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        "sleep 30",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    env._active_processes.add(process)
    await env.cleanup()
    assert env._active_processes == set()
    await process.wait()
