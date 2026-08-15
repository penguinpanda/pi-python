"""env.py 补充测试：路径解析、错误映射、超时与清理。"""

from __future__ import annotations

import asyncio
import os
import sys
import time
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


async def _wait_pid_dead(pid: int, timeout: float = 5.0) -> None:
    """轮询等待 pid 退出（含 init 收割僵尸的时间）。仅 POSIX。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"process {pid} still alive")


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
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    # Windows 的 expanduser 使用 USERPROFILE 而非 HOME。
    monkeypatch.setenv("USERPROFILE", str(home))

    tilde = await env.absolute_path("~/file.txt")
    assert tilde[0] is True
    assert tilde[1] == str(home / "file.txt")

    # 标准文件 URI（Windows 为 file:///C:/...，POSIX 为 file:///tmp/...）。
    file_url = await env.absolute_path((tmp_path / "a b" / "x.txt").as_uri())
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
    assert result[1].code == "callback_error"


@pytest.mark.skipif(sys.platform == "win32", reason="posix process-group kill only")
@pytest.mark.asyncio
async def test_exec_callback_error_kills_process_promptly(tmp_path: Path) -> None:
    """callback 抛错时 exec 应立即杀子进程并返回，而不是等命令自然结束。"""
    env = _env(tmp_path)
    pidfile = tmp_path / "pid.txt"

    def fail_callback(_chunk: str) -> None:
        raise RuntimeError("callback boom")

    started = time.monotonic()
    result = await env.exec(
        f"sleep 8 & echo $! > {pidfile}; echo marker; wait",
        ShellExecOptions(on_stdout=fail_callback),
    )
    elapsed = time.monotonic() - started
    assert result[0] is False
    assert result[1].code == "callback_error"
    assert elapsed < 4.0, f"exec waited for command to finish naturally ({elapsed:.1f}s)"
    pid = int(pidfile.read_text().strip())
    await _wait_pid_dead(pid)
    assert env._active_processes == set()


@pytest.mark.skipif(sys.platform == "win32", reason="posix process-group kill only")
@pytest.mark.asyncio
async def test_exec_cancellation_kills_process(tmp_path: Path) -> None:
    """外层任务被取消时 exec 必须杀子进程树并清理 _active_processes。"""
    env = _env(tmp_path)
    pidfile = tmp_path / "pid.txt"
    task = asyncio.create_task(env.exec(f"sleep 8 & echo $! > {pidfile}; wait"))
    for _ in range(100):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    assert pidfile.exists(), "shell did not start"
    pid = int(pidfile.read_text().strip())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_pid_dead(pid)
    assert env._active_processes == set()


@pytest.mark.asyncio
async def test_cleanup_kills_active_process(tmp_path: Path) -> None:
    env = _env(tmp_path)
    # 用当前解释器睡眠,避免依赖 sh(Windows 无 sh)。
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    env._active_processes.add(process)
    await env.cleanup()
    assert env._active_processes == set()
    await process.wait()
